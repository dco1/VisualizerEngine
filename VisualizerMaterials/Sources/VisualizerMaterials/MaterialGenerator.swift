import simd
import Foundation

/// Bakes procedural materials into tileable `MaterialChannels`. This is the
/// composable engine of MATERIALS_AND_TEXTURES §7 expressed on the CPU: each generator
/// is a small graph of `Noise` nodes → height → derived normal/curvature → roughness +
/// albedo. The terrazzo here is the §1 hero — the Roominate material that "turned out
/// well", generalized: seeded aggregate-chip scatter (Voronoi), per-chip palette +
/// roughness variation, a slightly-rougher cement binder, and a polish clearcoat.
public enum MaterialGenerator {

    /// Default bake resolution for every procedural material tile. 512 matches the
    /// engine atlas's slice size EXACTLY — the atlas letterboxes whatever it's given
    /// into a 512² slice, so a 256 bake was silently UPSCALED 2× on upload and every
    /// material rendered pre-blurred (Danny's "materials look like bitmaps"; the old
    /// hex anti-tiling dithered the blur, and turning it off for coherent wood/tile
    /// exposed it). Keep this in lockstep with the renderer's `atlasSliceSize`.
    public static let bakeSize = 512

    /// Sub-cell phase offset (in UV) applied to the aggregate-stone Voronoi samplers
    /// (granite, pebble) so the tile boundary falls at a grain/stone CENTRE rather than
    /// on the Voronoi cell lattice line. Both stones are already perfectly toroidal, but
    /// a lattice-aligned tile edge is a grain-boundary-dense column, so TextureAudit's
    /// seam ratio (edge diff ÷ mean interior diff) reads high even with no visible seam.
    /// 26 cells span the tile → half a cell is `0.5 / 26`; sampling there makes the wrap
    /// column representative of the interior (seam ≈ 1). Value is arbitrary within a cell;
    /// half-cell maximises distance from every lattice line.
    static let grainPhase = 0.5 / 26.0

    // MARK: – Phase 8 civil / site materials

    /// Grass ground cover: green base with height/density variation and stem detail.
    /// `height` controls blade height scale (0.04 = lawn, 0.12 = meadow, 0.18 = tall grass).
    public static func grass(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 71, height: Double = 0.08) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        let hScale = clamp01(height / 0.20)    // normalize height to albedo darkening
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Clump structure: coarse density patches (sparse → bare soil, dense → full grass).
                let clump = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: seed)
                // Per-blade micro variation: thin vertical striations simulate stems.
                let stem  = Noise.fbmTiled(u * 1.0, v * 20.0, baseCells: 4, octaves: 3, seed: seed ^ 0xB3)
                // Macro tonal variation (sun/shade patches).
                let shade = Noise.fbmTiled(u, v, baseCells: 1, octaves: 2, seed: seed ^ 0x29)

                // Albedo: grass green, darker in shadows and bare-soil gaps.
                let density = clamp01(clump * 1.4 - 0.1)   // 0 = bare soil, 1 = full grass
                let grassLum  = clamp01(0.25 + 0.12 * clump + 0.06 * stem - 0.05 * shade * hScale)
                let soilLum   = clamp01(0.20 + 0.08 * shade)
                let lum       = soilLum + (grassLum - soilLum) * density
                // Grass: strong green tint, varies from warm-green to cool-green by shade.
                let r = clamp01(lum * (0.48 - 0.06 * shade))
                let g = clamp01(lum * (1.00 + 0.02 * clump))
                let b = clamp01(lum * (0.38 + 0.04 * shade))
                ch.albedo[ch.idx(x, y)] = Vec3(r, g, b)

                // Roughness: high overall, slightly lower on flat lawn, higher in dense grass.
                ch.roughness[ch.idx(x, y)] = clamp01(0.75 + 0.15 * density + 0.08 * clump)

                // Height: blade tips above base — taller at clump centers.
                ch.height[ch.idx(x, y)] = clamp01(0.30 + 0.35 * density * hScale + 0.15 * stem)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 1.5)
        return ch
    }

    /// Manicured lawn / mowed turf — the Phase 8 yard default. Distinct from `grass` (tall,
    /// clumpy, soil-gapped): this is a dense, even green carpet with fine blade-tip micro-texture,
    /// broad sun/shade tonal mottle, and subtle drought/wear patches (a touch of yellow-brown).
    /// High roughness, varied spatially so it passes `TextureAudit` (#3). No clearcoat.
    ///
    /// FAR-FIELD BANDING FIX (2026-07): the mow-stripe (`cos(2π·stripeCells·v)`) is GONE. It was
    /// a COHERENT periodic feature — at uvScale 2 m it repeated as ruler bands every 0.5 m across
    /// the whole yard, and living in the HEIGHT/roughness channels it drove a periodic normal +
    /// specular ripple that read, at grazing angle under a low sun, as horizontal LOD-row banding
    /// in every wide outdoor shot (Danny's "far-field lawn banding" — a real-frame audit finding).
    /// Demoting its albedo amplitude (an earlier pass) didn't kill it because the ripple lived in
    /// the derived NORMALS. A coherent feature that tiles identically every 2 m is fundamentally
    /// incompatible with the hex-stochastic de-repeat (`.ground` opts IN) — the de-repeat only
    /// hides STOCHASTIC content. So the stripe is removed entirely and the turf variation is now
    /// fully stochastic (blade micro + broad macro mottle + drought), which the de-repeat hides:
    /// no coherent period survives to band. `stripeCells` is retained (source-compat) but unused.
    public static func lawn(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 23,
                            stripeCells: Int = 4, wear: Double = 0.22) -> MaterialChannels {
        _ = stripeCells   // retained for source-compat; the mow-stripe was removed (banding fix)
        var ch = MaterialChannels(size: size, category: .ground)
        let wear01 = clamp01(wear)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Fine blade-tip striations (micro mottle). Frequency pulled down from v·28 to
                // v·16 so the derived normal detail doesn't alias into glitter at grazing minification.
                let blade = Noise.fbmTiled(u * 2.0, v * 16.0, baseCells: 4, octaves: 3, seed: seed ^ 0xC7)
                // Broad STOCHASTIC sun/shade tonal mottle — replaces the mow-stripe's role of
                // large-scale variation, but with NO coherent period (the de-repeat hides it), so
                // the field reads as living turf instead of a flat sheet without re-introducing bands.
                let mottle = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: seed ^ 0x5A)
                // Macro health patches: lush vs. drought/wear (yellow-brown). Smooth + sparse —
                // broad soft patches, not high-frequency speckle (which reads as gravel/litter).
                let patch = Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed ^ 0x91)
                let droughtFrac = clamp01((patch - 0.62) * 1.2) * wear01   // 0 lush … >0 stressed

                // Base green intensity: blade micro + broad tonal mottle + mild wear darkening.
                // All terms stochastic → no ruler-band period.
                let gLevel = clamp01(0.42 + 0.10 * blade + 0.06 * (mottle - 0.5) - 0.06 * droughtFrac)
                // REAL turf albedo, not stadium-paint green: G leads but tops out ~0.25
                // (a lawn is a dark surface — measured turf reflectance is 8–25%). The old
                // G≈0.5–0.7 rendered as saturated flat "bitmap" green at any distance.
                //
                // DESATURATION PASS (2026-07): with the mow-stripe removed the field read as a
                // uniform, over-saturated pure Kelly-green sheet (albedo ≈(0.09,0.27,0.06),
                // G−R≈0.17). Real turf is a muted, slightly yellow/olive-leaning green with much
                // less pure-green chroma. So R and B are lifted toward G (R more than B → a touch
                // of yellow) — the green-lead gap G−R shrinks ~0.17→~0.11 and G−B ~0.19→~0.16 while
                // the surface stays unambiguously GREEN. A low-freq olive modulation adds subtle
                // yellow-olive patchiness (broad, not per-pixel) so it's not a flat sheet without
                // adding coherent-period banding or spiking the mowed-lawn's low chroma variance.
                //
                // CARPET PASS (DH-0077, 2026-09-13): the muted olive left the lawn's green to the
                // blades, and from a dollhouse camera those read as "teeny little hairs" on a dull
                // ground (Danny). A mowed lawn seen from above resolves as ONE green surface, so the
                // tile now carries a richer turf green — G ≈0.23→0.28 at the mean, B pulled back
                // under it (G−R ≈0.09→0.15) — still well inside measured turf reflectance and the
                // dielectric band, and the blades are matched to THIS colour rather than out-running it.
                // (A first cut at G−R ≈0.19 rendered as lime — the neon read 7a08a96 removed.)
                let olive = 0.02 * (mottle - 0.5)   // ±0.01 low-freq yellow-olive patch tint on R
                let lush = Vec3(clamp01(0.085 + gLevel * 0.12 + olive),
                                clamp01(0.09 + gLevel * 0.46),
                                clamp01(0.038 + gLevel * 0.06))
                // Stressed patches yellow-brown (straw): R rises toward G, B stays low.
                let straw = Vec3(clamp01(0.10 + gLevel * 0.55),
                                 clamp01(0.08 + gLevel * 0.62),
                                 clamp01(0.05 + gLevel * 0.18))
                let albedo = lush + (straw - lush) * droughtFrac
                ch.albedo[ch.idx(x, y)] = albedo

                // Roughness: high turf scatter, higher on dry wear, spatially varied by the
                // STOCHASTIC blade micro + broad mottle (centred so they spread both ways). This
                // carries the roughness std-dev TextureAudit (#3) requires — the removed mow-stripe
                // used to supply it via `-0.06·band`, so the stochastic terms are widened to match
                // WITHOUT re-introducing a coherent period.
                ch.roughness[ch.idx(x, y)] = clamp01(0.76 + 0.10 * (blade - 0.5) + 0.06 * (mottle - 0.5) + 0.06 * droughtFrac)

                // Height: blade-tip micro-relief only — no coherent ridge (the stripe ridge was
                // the normal-ripple banding source).
                ch.height[ch.idx(x, y)] = clamp01(0.40 + 0.20 * blade)
            }
        }
        ch.clearcoat = 0.0
        // Softer normal derivation (was 1.6): with the coherent stripe ridge gone, less relief
        // amplification keeps the remaining stochastic micro-relief from aliasing at grazing angle.
        ch.deriveNormals(strength: 1.3)
        return ch
    }

    /// The per-species knobs over the ONE clipped-hedge foliage generator (`hedgeFoliage`).
    /// Species-based hedges (DH-0123): a real boxwood, privet and holly are the same *surface
    /// kind* — a dense stochastic Worley leaf lattice — but differ in leaf colour, leaf/spray
    /// scale, how much bright new growth flushes, and how waxy (glossy) the lamina reads. Only
    /// these knobs vary; the noise structure is shared so all three tile seamlessly and pass the
    /// same `TextureAudit`. Phase A is BROADLEAF only — needled conifers (yew, arborvitae) need
    /// geometric needle cards (Phase B), not a material tint.
    public struct HedgeFoliageParams: Sendable {
        /// Deep shaded-lamina leaf green, and the lighter tint of sun-lit fresh growth.
        public var deep: Vec3
        public var fresh: Vec3
        /// Voronoi cell counts — the broad leaf-spray scale and the fine individual-leaf scale.
        /// Fewer cells → larger leaves (privet); more → smaller, denser leaves (holly).
        public var coarseCells: Int
        public var fineCells: Int
        /// Multiplier on the fresh-growth fraction: 1 = boxwood's yellow-green flush, <1 keeps a
        /// species uniformly dark (holly), >1 flushes brighter (privet).
        public var newGrowth: Double
        /// Base roughness of the leaf face — lower is glossier/waxier.
        public var roughnessBase: Double
        /// Clearcoat leaf sheen — the waxy highlight, not a wet gloss.
        public var clearcoat: Double

        public init(deep: Vec3, fresh: Vec3, coarseCells: Int = 11, fineCells: Int = 22,
                    newGrowth: Double = 1.0, roughnessBase: Double = 0.52, clearcoat: Double = 0.08) {
            self.deep = deep; self.fresh = fresh
            self.coarseCells = coarseCells; self.fineCells = fineCells
            self.newGrowth = newGrowth; self.roughnessBase = roughnessBase; self.clearcoat = clearcoat
        }

        /// Clipped **boxwood** — deep green, small waxy leaves, a modest yellow-green new-growth
        /// flush. The shipped default (DH-0122); these values reproduce the original `boxwood`
        /// generator exactly, so the boxwood material is byte-identical to before DH-0123.
        public static let boxwood = HedgeFoliageParams(
            deep: Vec3(0.045, 0.115, 0.040), fresh: Vec3(0.150, 0.250, 0.075),
            coarseCells: 11, fineCells: 22, newGrowth: 1.0, roughnessBase: 0.52, clearcoat: 0.08)

        /// **Privet** (Ligustrum) — a brighter, more open mid-green with LARGER oval leaves and a
        /// vigorous bright new-growth flush; glossier than boxwood. PROVISIONAL (DH-0123 Phase A):
        /// tuned from reference, to be judged on the published hero capture, not frozen here.
        public static let privet = HedgeFoliageParams(
            deep: Vec3(0.055, 0.140, 0.050), fresh: Vec3(0.175, 0.300, 0.095),
            coarseCells: 8, fineCells: 16, newGrowth: 1.35, roughnessBase: 0.46, clearcoat: 0.12)

        /// **Holly** (Ilex) — very dark blue-green, small dense spiny leaves, almost no bright
        /// flush (it stays uniformly dark), and a hard waxy gloss. PROVISIONAL (DH-0123 Phase A):
        /// tuned from reference, to be judged on the published hero capture, not frozen here.
        public static let holly = HedgeFoliageParams(
            deep: Vec3(0.028, 0.082, 0.046), fresh: Vec3(0.068, 0.145, 0.072),
            coarseCells: 13, fineCells: 28, newGrowth: 0.35, roughnessBase: 0.40, clearcoat: 0.17)
    }

    /// Clipped **hedge foliage** — a dense mass of small waxy leaves read at close range, NOT
    /// turf. The hedge placeable used the `grass` tile as a stand-in, but grass is soil-gapped
    /// vertical blades: it reads as lawn stood on end, never as a trimmed shrub. A clipped hedge
    /// is the opposite surface — a continuous canopy of overlapping oval leaves with dark recesses
    /// between the sprays, deep green shading to a lighter tint where fresh growth catches the
    /// sun, and a faint waxy leaf sheen. All structure is a Worley leaf lattice + tiled fbm
    /// undulation, so it is fully stochastic and tileable — no coherent period to band in a wide
    /// yard shot — and `.ground` category (yard-only + the hex de-repeat, exactly like `grass`).
    ///
    /// The leaf relief lives ENTIRELY in the material (the hedge mesh is a clipped box, no
    /// per-leaf geometry), so this is a `.needsMicroRelief` surface — it ships a detail normal.
    /// Species differ only through `HedgeFoliageParams` (DH-0123).
    public static func hedgeFoliage(_ p: HedgeFoliageParams,
                                    size: Int = MaterialGenerator.bakeSize,
                                    seed: UInt64 = 47) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        let deep  = p.deep
        let fresh = p.fresh
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Broad clipped-canopy undulation — the shadow pockets between leaf sprays.
                let spray = Noise.fbmTiled(u, v, baseCells: 4, octaves: 4, seed: seed)
                // DOMAIN WARP before the leaf lattice. A raw Worley partition reads as regular
                // stone crazing (a continuous even outline, the mosaic tell); warping the sample
                // point with a low-freq tiled fbm buckles the cells into irregular organic
                // clusters. The warp field is periodic, so the wrap stays seamless.
                let wu = u + 0.05 * (Noise.fbmTiled(u, v, baseCells: 5, octaves: 2, seed: seed ^ 0x1A2B) - 0.5)
                let wv = v + 0.05 * (Noise.fbmTiled(u, v, baseCells: 5, octaves: 2, seed: seed ^ 0x3C4D) - 0.5)
                // TWO leaf scales so cluster size varies (a single scale is what makes a mosaic):
                // broad sprays + the fine individual leaves within them.
                let coarse = Noise.voronoiTiled(wu, wv, cells: p.coarseCells, jitter: 0.95, seed: seed ^ 0x5C2D)
                let fine   = Noise.voronoiTiled(wu, wv, cells: p.fineCells, jitter: 0.95, seed: seed ^ 0x77E9)
                let dome  = clamp01(1.0 - fine.f1 * 1.4)                    // 1 at a leaf centre
                let broad = clamp01(1.0 - coarse.f1 * 1.2)                  // 1 at a spray centre
                // Per-leaf tint, biased toward its broad cluster so whole sprays vary together.
                let leafT = mix(Noise.unit(fine.cellId), Noise.unit(coarse.cellId), 0.4)
                // Within-leaf micro tonal variation (waxy highlight vs. shaded lamina).
                let micro = Noise.fbmTiled(u * 6.0, v * 6.0, baseCells: 8, octaves: 2, seed: seed ^ 0x91A3)

                // A fraction of leaves are fresh new growth, biased toward the sun-lit sprays; the
                // rest stay deep green. `newGrowth` scales how much a species flushes.
                let freshFrac = clamp01((leafT - 0.50) * 1.6) * clamp01(0.4 + spray) * p.newGrowth
                var col = mix(deep, fresh, clamp01(freshFrac))
                // BROKEN gap: the leaf-edge outline only darkens where a micro-noise agrees, so
                // the shadow between leaves reads as intermittent pockets, not a crackle net.
                let edge = 1.0 - smoothstep(0.0, 0.10, fine.f2 - fine.f1)
                let pocket = edge * smoothstep(0.35, 0.70, micro)
                let shade = clamp01(0.26 * (1.0 - spray) + 0.42 * pocket)   // 0 lit … 1 recess
                col = col * (0.82 + 0.30 * micro) * (1.0 - 0.50 * shade)
                ch.albedo[ch.idx(x, y)] = clampBand(col)

                // Waxy leaf faces read semi-glossy; the shaded recesses go matte. The spatial
                // swing carries the roughness-std TextureAudit tell.
                ch.roughness[ch.idx(x, y)] = clamp01(p.roughnessBase + 0.30 * shade + 0.08 * (micro - 0.5))

                // Leaf clusters stand proud, recesses sink — the macro relief that reads as
                // many small overlapping leaves once deriveNormals runs.
                ch.height[ch.idx(x, y)] = clamp01(0.42 + 0.30 * dome + 0.18 * broad
                                                  + 0.14 * spray - 0.22 * pocket)
            }
        }
        ch.clearcoat = p.clearcoat   // faint waxy leaf sheen — not a wet gloss
        ch.deriveNormals(strength: 2.6)
        // Fine leaf-lamina tooth (the macro leaves are in the height field above; this is the
        // sub-leaf grain that stops the canopy reading plastic at a grazing angle).
        addMicroDetail(&ch, seed: seed ^ 0xB6, baseCells: 110, strength: 0.40)
        return ch
    }

    /// Clipped **boxwood** hedge foliage — the shipped default (DH-0122). Delegates to the shared
    /// `hedgeFoliage` generator with the boxwood species knobs; byte-identical to the pre-DH-0123
    /// implementation.
    public static func boxwood(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 47) -> MaterialChannels {
        hedgeFoliage(.boxwood, size: size, seed: seed)
    }

    /// Clipped **privet** hedge foliage — a brighter, larger-leaved broadleaf species (DH-0123).
    public static func privet(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 47) -> MaterialChannels {
        hedgeFoliage(.privet, size: size, seed: seed)
    }

    /// Clipped **holly** hedge foliage — a very dark, small-leaved, hard-waxy broadleaf species
    /// (DH-0123).
    public static func holly(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 47) -> MaterialChannels {
        hedgeFoliage(.holly, size: size, seed: seed)
    }

    /// Asphalt road surface: near-black base aging to worn gray; Voronoi aggregate pitting;
    /// roughness 0.82–0.95 spatially varied; micro-crack scatter via sparse fbm.
    public static func asphalt(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 61, age: Double = 0.4) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        let age01 = clamp01(age)   // 0 = fresh black, 1 = worn gray
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Voronoi for aggregate pitting — each cell centre is a pit.
                let cell = Noise.voronoiTiled(u, v, cells: 18, seed: seed)
                let pit = clamp01(1.0 - cell.f1 * 2.5)   // bright at cell edge, dark in center
                // Large-scale macro tone variation (weathering patches).
                let macro = Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed ^ 0x11)
                // Micro-crack scatter — sparse, thin, dark lines.
                let micro = Noise.fbmTiled(u, v, baseCells: 12, octaves: 2, seed: seed ^ 0x33)
                let crack = clamp01(micro - 0.45) * 2.0   // sparse dark filaments

                // Albedo: fresh near-black (0.05) aging toward worn gray (0.30).
                let freshBase = 0.05 + 0.04 * macro
                let wornBase  = 0.25 + 0.05 * macro
                var lum = freshBase + (wornBase - freshBase) * age01
                lum += 0.03 * pit          // slight aggregate spec at cell edges
                lum -= 0.015 * crack       // cracks are darker
                lum = clamp01(lum)
                ch.albedo[ch.idx(x, y)] = Vec3(lum, lum * 0.99, lum * 0.97)

                // Roughness: very rough (0.88–0.95 worn, 0.82–0.90 fresh); pits add roughness.
                let roughBase = 0.82 + 0.06 * age01
                ch.roughness[ch.idx(x, y)] = clamp01(roughBase + 0.07 * pit + 0.04 * macro)

                // Height: pitting at Voronoi edges; macro undulation; cracks are low.
                ch.height[ch.idx(x, y)] = clamp01(0.50 + 0.18 * macro + 0.10 * pit - 0.08 * crack)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 1.4)
        // Coarse aggregate grit — asphalt is a very rough sand/stone matrix; without the
        // fine tooth it reads as a flat black plastic sheet at grazing angles.
        addMicroDetail(&ch, seed: seed ^ 0xB6, baseCells: 90, octaves: 2, strength: 0.80)
        return ch
    }

    /// Exterior sidewalk concrete: light gray broom finish (directional striations in
    /// roughness + height), outdoor weathering (mossy-edge staining, no clearcoat).
    public static func sidewalkConcrete(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 67) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Broom finish: directional striations via integer v-scale (40×) — tile-safe.
                // baseCells 4 ensures enough structure at test size 96; integer v*40 gives
                // 40 complete periods in v so the wrap is exact at period 1/40.
                let broom = Noise.fbmTiled(u, v * 40.0, baseCells: 4, octaves: 4, seed: seed)
                // Large-scale weathering patches (macro luma variation — must pass macroCV gate).
                let weather = Noise.fbmTiled(u, v, baseCells: 2, octaves: 4, seed: seed ^ 0x77)
                // Fine aggregate show (micro stipple adds to roughness variance).
                let agg = Noise.fbmTiled(u, v, baseCells: 18, octaves: 2, seed: seed ^ 0xAA)

                // Base albedo: light gray with substantial weathering range to pass macroCV.
                // clean ≈ 0.72, stained ≈ 0.50 → ~30% patches give sufficient luma swing.
                let stainFrac = clamp01((weather - 0.2) * 0.7)  // stronger stain coverage
                // DH-1397: weathered pavement concrete reads 0.30-0.55 (was clean 0.72 / stained 0.48 ⇒ mean 0.70).
                let cleanLum  = clamp01(0.52 + 0.04 * broom)
                let stainLum  = clamp01(0.34 + 0.05 * weather)
                let lum = cleanLum + (stainLum - cleanLum) * stainFrac
                // Staining gives slight greenish cast.
                ch.albedo[ch.idx(x, y)] = Vec3(clamp01(lum - 0.010 * stainFrac),
                                               clamp01(lum + 0.015 * stainFrac),
                                               clamp01(lum - 0.018 * stainFrac))

                // Roughness: substantial variation (broom high-frequency × weather × agg).
                // Must pass roughnessStdDev > 0.012; the range 0.60–0.90 guarantees it.
                ch.roughness[ch.idx(x, y)] = clamp01(0.60 + 0.20 * (1.0 - broom) + 0.10 * agg + 0.06 * weather)

                // Height: broom ridges and aggregate.
                ch.height[ch.idx(x, y)] = clamp01(0.50 + 0.28 * broom + 0.12 * agg + 0.10 * weather)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 2.0)
        // Fine aggregate/broom grit — the exposed sand in the broom finish. Rough concrete
        // needs the micro-tooth or it flattens to plastic under grazing sun.
        addMicroDetail(&ch, seed: seed ^ 0xB7, baseCells: 90, octaves: 2, strength: 0.70)
        return ch
    }

    // ── The YARD's run, and the band table every ground generator is written against ──
    //
    /// **World metres one ground bake spans.** The yard's generators reason in real
    /// centimetres and the surface's UV period is taken FROM the sizes they chose — the same
    /// inversion tile and terrazzo use, and the reason those two are the materials Danny has
    /// called good. Changing this re-scales every band below; treat the two as ONE decision.
    ///
    /// **2.0 m, and the 1.0 m this briefly shipped at was a mistake — measured, not argued.**
    /// The reasoning for going finer was that 1.0 m halves the texel (3.91 mm → 1.95 mm) and is
    /// the finest run that still meets the per-vertex world tint, whose floor is two terrain
    /// cells (`TerrainMesh.focusCellMeters` = 0.5 m). Both halves of that are true and neither
    /// mattered, because **the camera cannot use the texels it buys**:
    ///
    /// | run | delivered grain, forest floor | delivered grain, dirt | recurrences per 512 px |
    /// |---|---|---|---|
    /// | 1.0 m | 8.4 % | 7.9 % | **6** |
    /// | 2.0 m | 8.4 % | 7.4 % | **2** |
    ///
    /// (Grain is the 2–16 px contrast on the clean yard square; recurrences are 40×40 patches
    /// matching elsewhere in the crop at ≥ 0.60 NCC. Both measured on the `4000 Sunset`
    /// dollhouse cutaway, hex de-repeat on, same generator, only this constant changed.)
    ///
    /// The grain is identical because the yard renders at ~6 mm per pixel near-field and ~13 mm
    /// mid-frame, so everything the eye resolves is ≥ 12 mm — comfortably above the 7.8 mm a
    /// 3.91 mm texel can represent. The extra resolution lands entirely below the camera, which
    /// is the SAME argument that retired `MaterialScaleAudit`'s 3 mm touch band for this shot,
    /// applied one level up. What it does cost is repetition: halving the run quadruples the
    /// repeats per unit area, and the hex de-repeat can shuffle WHERE a tile is sampled but
    /// cannot invent new content — so the same clump turned up three times as often.
    ///
    /// The band table below is written in physical sizes and re-derived for this run, so it is
    /// unchanged in millimetres. The one band that genuinely wanted the finer run is `fines`,
    /// and it is roughness/relief only, below what the camera resolves — see its note.
    public static let groundRunMeters: Double = 2.0

    /// The ground's band table, at `groundRunMeters`. Named rather than typed inline so a run
    /// change re-scales one table instead of a dozen literals, and so the physical size each
    /// band lands at is written down where the generator can be read.
    ///
    /// | band | cells | feature @ 1.0 m | carries |
    /// |---|---|---|---|
    /// | `patchCells` | 4 | 50 cm | damp/dry tone — *pattern*, deliberately a whisper |
    /// | `aggregateCells` | 17 / 37 / 73 | 11.8 / 5.4 / 2.7 cm | the clumps — see `earth` |
    /// | `warpCells` | 52 | 3.8 cm | the domain warp that bends a clump out of a disc |
    /// | `pebbleCells` | 44 | 4.5 cm | stones |
    /// | `gritCells` | 144 | 1.4 cm | grit — tone + roughness breakup |
    /// | `finesCells` | 224 | 8.9 mm | fines — roughness + relief only (2.3 texels: at Nyquist) |
    ///
    /// The detail band is not in the table because it does not get its own noise: it is
    /// derived from `finesCells` and the litter mat, so the sub-millimetre relief and the macro
    /// channels describe ONE surface (see `setDetailRelief`).
    enum GroundBands {
        static let patchCells = 4
        /// **Three aggregate scales, and all three counts are PRIME on purpose.** 17 / 37 / 73
        /// share no common factor, so the three lattices never come back into register and the
        /// stack has no repeat finer than the tile itself. Even ratios (18 / 38 / 74 — the
        /// obvious doubling of the previous 9 / 19 / 37) share a factor of 2 and line up every
        /// other cell, printing a coarse grid over the whole yard.
        static let aggregateCells = [17, 37, 73]
        /// The shared domain-warp field's cell count, and how far it displaces a lookup,
        /// in CELLS of the layer being warped. See `earth` — the warp is what turns a Voronoi
        /// island from a disc into a clump, so these two are look decisions, not tuning.
        static let warpCells = 52
        static let warpCellFraction = 0.55
        static let gritCells = 144
        /// **At the Nyquist limit, and that is the ceiling rather than a choice.** At
        /// `groundRunMeters` a texel is 3.91 mm, so the finest representable feature is 7.8 mm;
        /// 224 cells puts this band at 8.9 mm, just above it. Asking for more is not a finer
        /// surface, it is aliasing — and this band carries no TONE for exactly that reason.
        static let finesCells = 224
        /// Stones, as a Voronoi cell count. 44 ⇒ ~4.5 cm, ~1 cell in 3 carrying a stone: a
        /// DENSE field, deliberately. A sparse landmark grids at the tile period and reads as a
        /// stamped repeat — see [[daydream-landmarks-cannot-be-baked]].
        static let pebbleCells = 44
    }

    // ── EARTH: the shared bed under every bare-ground yard ────────────────────────────
    //
    /// **A soil palette.** Bare dirt and a forest floor are the same surface with different
    /// mineral colour and a different amount of organic matter lying on it, so they are ONE
    /// generator with two palettes rather than two generators that drift apart
    /// ([[feedback-unify-sibling-entities]]). Before 2026-08-20 `forestFloor` was not a forest
    /// floor at all — `SurfaceDefault.groundRequest` aliased it to `dirt(damp: 0.55)` with a
    /// comment promising a real one later.
    public struct EarthPalette: Sendable, Equatable {
        /// Sun-baked high ground — the pale end of the mineral soil.
        public var dry: Vec3
        /// Moist hollow — the dark end. Wet soil is darker AND less saturated per unit value,
        /// because the water film specularly reflects the sky rather than scattering.
        public var damp: Vec3
        /// The mineral grit and small stones pressed into the surface. Greyer than the soil:
        /// they are rock, not humus, and a pebble that is merely a lighter version of the soil
        /// reads as a bald patch instead of a stone.
        public var grit: Vec3
        /// Open-surface roughness. Soil is very rough; the number varies spatially below.
        public var roughness: Double

        public init(dry: Vec3, damp: Vec3, grit: Vec3, roughness: Double) {
            self.dry = dry; self.damp = damp; self.grit = grit; self.roughness = roughness
        }
    }

    /// **Bare earth, authored band by band.** The generator behind `dirt` and `forestFloor`.
    ///
    /// Danny, 2026-08-20, on a dollhouse export of `4000 Sunset`: the exterior "doesn't look
    /// photographic or realistic at all", and the yard "has no fine grain at all". Measured on
    /// that frame before any of this was written, the yard's residual luma SD ran **1.0–1.6 out
    /// of 255 at every band from 4 to 32 px**, against a mean of 140 — about 1 % contrast at
    /// every size a pixel can resolve, with the energy climbing monotonically toward the
    /// coarsest band. That is the numeric signature of a smooth wash with blobs on it.
    ///
    /// **The defect was never resolution.** At the old 2 m run a texel was 3.9 mm, so the whole
    /// 2–50 cm band a dollhouse camera resolves was already representable — no amplitude had
    /// been put there. The proof was in the same library: `stone` and `pebble` measured 20–28 %
    /// albedo contrast in that band and read as real surfaces, `dirt` measured **3.1 %**.
    ///
    /// **Two wrong turns, both worth recording, because the bake shows each in one look and a
    /// GPU frame does not** (`GroundSwatchDumpTests` exists so the next person gets that look):
    ///
    ///  1. **A cellular PARTITION draws crazing, not clumps.** The obvious move is a Voronoi
    ///     partition darkened at its boundaries (`f2 − f1` → 0). But that boundary set is a
    ///     **connected, space-filling web by construction**, so it does not draw lumps — it
    ///     draws a crack network. Two of them at two cell sizes nests one web inside the other,
    ///     and the yard came back as dried lakebed.
    ///  2. **Pure fbm has no edges at any amplitude.** Turning the old generator's 1 % up to
    ///     10 % removed the "no detail" complaint and replaced it with watercolour: fbm is a sum
    ///     of smoothly interpolated lattices, so every feature it draws has a soft shoulder and
    ///     the surface reads as a wash however loud it gets. Amplitude was never the whole story.
    ///
    /// **What earth actually is, and what this draws: a PILE of overlapping clumps.** Three
    /// layers of *islands* — one seed per Voronoi cell, kept only where a per-cell roll says so,
    /// each a rounded lump with its own tone — at 11 / 5.3 / 2.7 cm. Islands do not tile the
    /// plane, so nothing draws a boundary web; they overlap, so the surface has genuine EDGES
    /// where one clump laps another; and the three cell counts are mutually prime, so the
    /// lattices never come back into register. Underneath sits an fbm bed for the continuity a
    /// pile of discrete objects would otherwise lack, and the same stack drives relief, so the
    /// macro normal and the tone agree about where the surface is broken.
    ///
    /// This is the shape `pebble` already had and got 20 % contrast from — per-object tonal
    /// variance is what carries a ground material, and `pebble`'s gap-shadow is a modest −0.16
    /// on top. Gravel is genuinely a packed mosaic so a partition suits it; soil is a pile, so
    /// it gets islands.
    ///
    /// The moisture patch, the one band that was already loud, is turned DOWN: at this camera it
    /// is *pattern* rather than texture, and it is what "blobby" was naming.
    ///
    /// `litter` is the organic overlay: 0 is bare mineral soil, 1 a closed leaf mat. It is a
    /// coverage fraction, not a colour — the leaves' own tone comes from `litterPalette`.
    ///
    /// **Cost, measured** (release, 512², 2026-08-20): 152 ms for `dirt` and 171 ms for
    /// `forestFloor`, against **61 ms** for the generator this replaced — four Voronoi lookups
    /// and a warp field per texel is where it goes. That is paid ONCE per finish: the ground is
    /// cached like every other material, and `MaterialResolver.prewarm` bakes all of
    /// `GroundMaterial.allCases` off the render thread at launch, across cores. If a future
    /// change puts a ground bake back on the render thread it will be felt — see
    /// [[daydream-placement-hang-material-library]], which is that bug at 512² already.
    static func earth(size: Int, seed: UInt64, damp: Double, palette: EarthPalette,
                      litter: Double = 0, litterPalette: (Vec3, Vec3) = (Vec3(0.30, 0.20, 0.10),
                                                                         Vec3(0.46, 0.33, 0.16)))
    -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        let damp01 = clamp01(damp)
        let litter01 = clamp01(litter)
        // The detail band's height field, accumulated as we go so the sub-millimetre grit is
        // derived from the SAME surface the macro channels describe rather than an unrelated
        // second noise (one relief, one writer — see `setDetailRelief`).
        var micro = [Double](repeating: 0, count: size * size)

        // Per-layer amplitude. Falls with cell size, but slowly — a steep falloff is what puts
        // all the energy in the coarsest layer and reads as blobs. The 12.5 cm layer is the one
        // a dollhouse camera resolves as texture; the 2.7 cm layer is what it resolves as grain.
        let layerTone = [0.52, 0.40, 0.28]
        let layerRelief = [0.26, 0.17, 0.11]
        // How many cells carry a clump. Below ~1 the gaps between clumps become the subject and
        // the surface reads as scattered objects on a plate rather than as broken ground.
        let layerDensity = [0.80, 0.78, 0.72]

        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)

                // ── 50 cm — moisture. PATTERN, not texture: kept to a whisper on purpose.
                let moisture = Noise.fbmTiled(u, v, baseCells: GroundBands.patchCells,
                                              octaves: 3, seed: seed ^ 0x4C)
                let wetFrac = clamp01(moisture - 0.5 + damp01 * 0.4)

                // ── 1.4 cm — GRIT. Tone, and the roughness breakup that clears tell #3.
                let grit = Noise.fbmTiled(u, v, baseCells: GroundBands.gritCells,
                                          octaves: 2, seed: seed ^ 0xA3) - 0.5
                // ── 6 mm — FINES. Three texels wide: near Nyquist, so relief and roughness
                // only. TONE this fine would alias into sparkle under minification.
                let fines = Noise.fbmTiled(u, v, baseCells: GroundBands.finesCells,
                                           octaves: 1, seed: seed ^ 0xC5)
                // ── 11 → 2.7 cm — the fbm BED under the clumps. Modest: it is the continuity
                // between them, not the structure, which the layers below supply.
                let bed = Noise.fbmTiled(u, v, baseCells: GroundBands.aggregateCells[0],
                                         octaves: 3, gain: 0.6, seed: seed ^ 0x9A) - 0.5

                // ── The DOMAIN WARP, and it is not a polish pass ──────────────────────────
                // An unwarped Voronoi island is a DISC: its seed is a point and `f1` is a
                // radius, so every clump comes out round and the surface reads as bokeh —
                // soap bubbles on a wash, which is what the first island attempt looked like.
                // Warping the lookup coordinate by a higher-frequency field before the distance
                // is taken bends those circles into the angular, amoeboid outlines broken earth
                // actually has, and it breaks the lattice's regular SPACING at the same time.
                //
                // ONE warp field, shared by all three layers and by the stones, scaled per
                // layer to about a quarter-cell of typical displacement. Shared on purpose: the
                // ground is deformed as a whole, so clumps at different scales that bend the
                // same way read as one surface — and it costs two fbm calls instead of eight.
                // `fbmTiled` is periodic, so the warped coordinate field is periodic too and
                // the tile still wraps.
                let warpX = (Noise.fbmTiled(u, v, baseCells: GroundBands.warpCells,
                                            octaves: 2, seed: seed ^ 0x1D) - 0.5) * 2
                let warpY = (Noise.fbmTiled(u, v, baseCells: GroundBands.warpCells,
                                            octaves: 2, seed: seed ^ 0x2E) - 0.5) * 2

                // ── The clump stack ──────────────────────────────────────────────────────
                var clumpTone = 0.0        // signed tonal contribution
                var clumpRelief = 0.0      // signed height contribution
                var topLayer = 0.0         // coverage of the FINEST clump present, for litter
                for (i, cells) in GroundBands.aggregateCells.enumerated() {
                    // `grainPhase` lands the tile boundary at a clump CENTRE rather than on the
                    // Voronoi lattice line, so the (already toroidal) wrap column is not a
                    // clump-boundary column — the trick `pebble` uses to hold its seam ratio
                    // near 1. Each layer is offset again by its index so the three stacks are
                    // not co-phased.
                    let phase = Self.grainPhase * Double(i + 1)
                    let w = GroundBands.warpCellFraction / Double(cells)
                    let c = Noise.voronoiTiled(u + phase + warpX * w, v + phase + warpY * w,
                                               cells: cells,
                                               jitter: 1.0, seed: seed &+ UInt64(i) &* 0x5D3F)
                    guard Noise.unit(Noise.mix(c.cellId ^ 0x9E)) < layerDensity[i] else { continue }
                    // The clump's own extent, jittered per clump, and a soft-but-DEFINED rim:
                    // the shoulder is ~35 % of the radius. Softer and the edge dissolves into
                    // the wash the fbm attempt produced; harder and the rim reads as a cut-out.
                    let r = 0.34 + 0.30 * Noise.unit(Noise.mix(c.cellId ^ 0x41))
                    let body = smoothstep(r, r * 0.80, c.f1)
                    guard body > 0 else { continue }
                    let value = Noise.unit(Noise.mix(c.cellId ^ 0x77)) - 0.5
                    clumpTone += layerTone[i] * value * body
                    // A dome, so the clump's crown catches light and its flank turns away.
                    clumpRelief += layerRelief[i] * body * (0.35 + 0.65 * smoothstep(r, 0, c.f1))
                    topLayer = Swift.max(topLayer, body)
                }

                // ── 4.5 cm — STONES, as islands of a different SUBSTANCE (mineral, not soil).
                let pw = GroundBands.warpCellFraction / Double(GroundBands.pebbleCells)
                let peb = Noise.voronoiTiled(u + Self.grainPhase + warpX * pw, v + warpY * pw,
                                             cells: GroundBands.pebbleCells,
                                             jitter: 1.0, seed: seed ^ 0x5D)
                let pebSize = 0.10 + 0.26 * Noise.unit(Noise.mix(peb.cellId ^ 0xB1))
                let pebble = Noise.unit(Noise.mix(peb.cellId)) < 0.34
                           ? smoothstep(pebSize, pebSize * 0.45, peb.f1) : 0.0
                let pebValue = Noise.unit(Noise.mix(peb.cellId ^ 0x3F)) - 0.5

                // ── Tone assembly ────────────────────────────────────────────────────────
                var col = palette.dry + (palette.damp - palette.dry) * wetFrac
                col *= 1.0 + clumpTone            // the clump stack — the band that reads
                col *= 1.0 + 0.20 * bed           // continuity between clumps
                col *= 1.0 + 0.16 * grit          // 1.4 cm
                // Stones: greyer than the soil, and lighter or darker per stone. Blended IN
                // rather than added, so a pale stone stays inside the dielectric band on pale
                // soil. A stone that is merely a lighter soil reads as a bald patch.
                let stone = palette.grit * (1.0 + 0.45 * pebValue)
                col = col + (stone - col) * (pebble * 0.80)

                // ── Organic litter, for the forest floor ─────────────────────────────────
                // Leaf flakes as ELONGATED islands — a leaf is not a disc, and a field of discs
                // reads as confetti (it did). Coverage is biased AWAY from the clump crowns:
                // a leaf blows off a high point and settles in the hollow beside it, which is
                // what makes the mat read as lying ON the soil rather than mixed into it.
                if litter01 > 0 {
                    let lw = GroundBands.warpCellFraction / Double(GroundBands.aggregateCells[1])
                    let leaf = Noise.voronoiTiledAniso(u + warpX * lw, v + warpY * lw * 0.5,
                                                       cellsX: GroundBands.aggregateCells[1],
                                                       cellsY: GroundBands.aggregateCells[1] * 2,
                                                       jitter: 1.0, seed: seed ^ 0xE7)
                    let hollow = clamp01(0.72 - 0.5 * topLayer)
                    let r = 0.30 + 0.34 * Noise.unit(Noise.mix(leaf.cellId ^ 0x11))
                    let body = smoothstep(r, r * 0.62, leaf.f1)
                    let cover = clamp01(body * (0.30 + 0.95 * litter01) * (0.45 + 0.9 * hollow))
                    // Per-leaf tone: fresh-fallen tan through rotted brown. The two ends are
                    // deliberately CLOSE — a wide spread reads as scattered confetti, not a mat.
                    let age = Noise.unit(Noise.mix(leaf.cellId ^ 0x77))
                    var leafCol = litterPalette.0 + (litterPalette.1 - litterPalette.0) * age
                    leafCol *= 1.0 + 0.26 * grit                     // vein / curl value break
                    // A leaf edge sits proud of the one under it, so it shades its neighbour.
                    leafCol *= 1.0 - 0.24 * smoothstep(r * 0.62, r, leaf.f1)
                    col = col + (leafCol - col) * cover
                    micro[ch.idx(x, y)] += 0.30 * cover
                    clumpRelief += 0.10 * cover
                }

                ch.albedo[ch.idx(x, y)] = Vec3(clamp01(max(col.x, 0.055)),
                                               clamp01(max(col.y, 0.048)),
                                               clamp01(max(col.z, 0.042)))

                // ── Roughness ────────────────────────────────────────────────────────────
                // Damp hollows read markedly smoother (the water film levels the surface),
                // stones smoother still, grit and fines rougher. The wide spatial swing clears
                // tell #3, and it is also real: soil roughness is a property of how broken the
                // surface is, which is exactly what these bands describe.
                ch.roughness[ch.idx(x, y)] = clamp01(
                    palette.roughness
                    - 0.20 * wetFrac
                    + 0.16 * grit
                    + 0.10 * (fines - 0.5)
                    + 0.10 * (clumpRelief - 0.25)
                    - 0.24 * pebble)

                // ── Height ───────────────────────────────────────────────────────────────
                // The same stack the tone describes, so the macro normal and the shading agree
                // about where the surface is broken.
                ch.height[ch.idx(x, y)] = clamp01(
                    0.34
                    + clumpRelief
                    + 0.10 * bed
                    + 0.08 * grit
                    + 0.18 * pebble
                    - 0.08 * wetFrac)

                // The detail band's own relief: the fines, plus the litter mat's edge.
                micro[ch.idx(x, y)] += fines
            }
        }
        ch.clearcoat = 0.0
        // 1.8, not the 2.6 the old flat-tone bake used. The macro relief now describes REAL
        // structure at 2.7–11 cm, and at 2.6 that structure lit by a low sun turns every clump
        // rim into a hard ridge — the yard's own version of the sponge tell.
        ch.deriveNormals(strength: 1.8)
        setDetailRelief(&ch, height: micro, strength: 0.85)
        return ch
    }

    /// Bare soil / dirt yard — exposed earth, not a road. Warm red-brown loam: broad damp/dry
    /// tonal patches, discrete clods at two aggregate scales, a dense scatter of small stones,
    /// and fine grit. Outdoor-grade — high roughness everywhere, NO clearcoat — and the colour
    /// sits firmly in the warm-earth family (R > G > B).
    ///
    /// See `earth` for the band table and for why the clods are cellular rather than fbm.
    public static func dirt(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 131,
                            damp: Double = 0.35) -> MaterialChannels {
        earth(size: size, seed: seed, damp: damp,
              palette: EarthPalette(dry: Vec3(0.46, 0.32, 0.21),
                                    damp: Vec3(0.24, 0.16, 0.10),
                                    grit: Vec3(0.44, 0.41, 0.37),
                                    roughness: 0.86))
    }

    /// **Forest floor — leaf litter over damp humus.** The ground `4000 Sunset` stands on, and
    /// until 2026-08-20 not a distinct material at all: `SurfaceDefault.groundRequest` aliased
    /// it to `dirt(damp: 0.55)` behind a comment calling the real one "a Phase-8 follow-up".
    ///
    /// What makes it a forest floor rather than a damp dirt is the ORGANIC MAT: a dense field
    /// of 3.8 cm leaf flakes, tan through rotted brown, pooling in the hollows of the soil
    /// beneath (leaves blow into cracks and stay) and each one edge-shadowed against the one
    /// below it. The soil showing through between them is cooler and darker than a sun-baked
    /// yard — humus under a canopy, not loam in the open.
    public static func forestFloor(size: Int = MaterialGenerator.bakeSize,
                                   seed: UInt64 = 99, litter: Double = 0.72) -> MaterialChannels {
        earth(size: size, seed: seed, damp: 0.55,
              palette: EarthPalette(dry: Vec3(0.30, 0.24, 0.17),
                                    damp: Vec3(0.16, 0.13, 0.10),
                                    grit: Vec3(0.36, 0.35, 0.32),
                                    roughness: 0.88),
              litter: litter,
              litterPalette: (Vec3(0.26, 0.18, 0.10), Vec3(0.50, 0.35, 0.17)))
    }

    /// **Garden mulch — a bed of shredded bark chips over dark planting soil.** The regional
    /// base-ground a garden plot wears (DH-0575). Built on the same `earth` construction as
    /// `forestFloor` — an organic mat lying ON broken earth — but the overlay is BARK, not fallen
    /// leaves, and that changes three things: the chips are warmer and redder (fresh shredded
    /// hardwood/cedar, not a year of mixed rotted leaf), their tone spread is TIGHTER (a bagged
    /// mulch is close to one colour where a forest floor is every autumn shade at once), and the
    /// coverage runs near-closed (a fresh mulch bed hides the soil). The earth beneath is a dark,
    /// damp PLANTING bed — potting soil under the chips, not sun-baked open-yard loam — so where
    /// the chips thin it reads as shadow, not as a bald dirt patch.
    ///
    /// The spatially-varying roughness comes for free from `earth`: damp hollows level smoother,
    /// grit/fines roughen, so `TextureAudit`'s roughness-SD tell is cleared the same way `dirt`
    /// and `forestFloor` clear it. Outdoor-grade, no clearcoat, `.ground` category.
    public static func mulch(size: Int = MaterialGenerator.bakeSize,
                             seed: UInt64 = 157) -> MaterialChannels {
        earth(size: size, seed: seed, damp: 0.5,
              palette: EarthPalette(dry: Vec3(0.26, 0.17, 0.11),
                                    damp: Vec3(0.13, 0.085, 0.055),
                                    grit: Vec3(0.30, 0.24, 0.19),
                                    roughness: 0.90),
              litter: 0.92,
              litterPalette: (Vec3(0.34, 0.17, 0.10), Vec3(0.50, 0.28, 0.15)))
    }

    /// Phase 8 — tree BARK for the yard trunk/branches. Warm grey-brown with the canonical
    /// bark tells: near-vertical furrows (stretched fbm along the tile's V axis = the trunk's
    /// vertical), a knotty macro mottle (lighter ridges / darker fissures), and a matte, very
    /// rough finish that varies strongly ridge↔fissure so it clears `TextureAudit` tell #3.
    /// No clearcoat (bark is bone-dry). `.wood` category (dielectric).
    public static func bark(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 401) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .wood)
        // Warm bark endpoints: pale weathered ridge → dark damp fissure.
        let ridge = Vec3(0.34, 0.27, 0.20)
        let fissure = Vec3(0.13, 0.10, 0.075)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Vertical furrows running UP the trunk. A cosine over an INTEGER number of
                // furrows wraps seamlessly in u (period 1/furrows divides 1.0), and its phase
                // is wobbled by seamless fbm so the fissures meander instead of ruling straight.
                // This keeps the directional (vertical) furrow look WITHOUT a horizontal seam
                // (pre-multiplying u/v into fbm would break the toroidal wrap — the tell).
                let furrows = 16.0
                let wobble = Noise.fbmTiled(u, v, baseCells: 4, octaves: 3, seed: seed ^ 0x2F) - 0.5
                let furrow = 0.5 + 0.5 * cos(2.0 * Double.pi * furrows * u + wobble * 2.4)
                // Knotty macro patches (broad lighter/darker regions — the bark's character).
                let knot = Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed ^ 0x5B)
                // Fine grit for micro roughness variation (isotropic, seamless).
                let grit = Noise.fbmTiled(u, v, baseCells: 40, octaves: 2, seed: seed ^ 0xA7)

                // Fissure depth: where furrow is low we sit deep in a crack (dark), high = ridge.
                let ridgeFrac = clamp01(furrow * 1.2 - 0.1 + 0.15 * (knot - 0.5))
                var col = fissure + (ridge - fissure) * ridgeFrac
                col += Vec3(0.05, 0.04, 0.03) * (knot - 0.5)             // macro tone
                ch.albedo[ch.idx(x, y)] = Vec3(clamp01(max(col.x, 0.05)),
                                               clamp01(max(col.y, 0.04)),
                                               clamp01(max(col.z, 0.03)))

                // Roughness: very rough overall; ridges (weathered) a touch smoother than the
                // damp fissures, grit adds spatial swing. Wide range → clears tell #3.
                ch.roughness[ch.idx(x, y)] = clamp01(0.82 - 0.14 * ridgeFrac + 0.12 * grit)

                // Height: ridges stand proud, fissures recess — the furrow relief.
                ch.height[ch.idx(x, y)] = clamp01(0.30 + 0.50 * ridgeFrac + 0.14 * grit)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 3.0)     // deep bark relief
        // Bark grain runs vertically up the trunk (the furrows). A coherent vertical grain
        // tangent gives the anisotropic-highlight lobe its axis and satisfies the wood-grain
        // audit (a `.wood` material must carry a grain direction).
        ch.grainTangent = Array(repeating: Vec2(0, 1), count: size * size)
        return ch
    }

    /// Phase 8 — tree FOLIAGE for the yard canopy. A lush, saturated leaf-green with
    /// per-clump value/hue jitter (light sun-lit crown vs. shaded interior), a leafy micro
    /// mottle (so it doesn't read as a flat green ball), and a soft-matte finish. Green must
    /// out-run red by a wide margin so it stays green under the renderer's warm daytime white
    /// balance (same lesson as `lawn`). No clearcoat. `.ground` category (vegetation).
    public static func foliage(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 517) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Clump patches: sun-lit outer crown (lighter/warmer) vs shaded pockets (darker).
                let clump = Noise.fbmTiled(u, v, baseCells: 3, octaves: 4, seed: seed)
                // Leafy micro mottle — many small leaf-scale value steps.
                let leaf  = Noise.fbmTiled(u * 6.0, v * 6.0, baseCells: 6, octaves: 3, seed: seed ^ 0x3D)
                // A few autumn/dry-leaf flecks (sparse warmer specks) for life.
                let fleck = Noise.fbmTiled(u * 4.0, v * 4.0, baseCells: 5, octaves: 2, seed: seed ^ 0x91)

                let value = clamp01(0.55 + 0.30 * clump + 0.14 * leaf - 0.10 * (1 - clump))
                // Lush leaf green: G dominates; R/B kept low so it survives the warm grade.
                var r = value * (0.30 + 0.18 * clump)
                var g = value * (0.92 + 0.06 * leaf)
                let b = value * (0.22 + 0.06 * clump)
                // Sparse warmer flecks (a few sun-dried / autumn leaves): lift R where fleck peaks.
                let warm = clamp01((fleck - 0.72) * 2.4)
                r += warm * 0.22; g -= warm * 0.06
                ch.albedo[ch.idx(x, y)] = Vec3(clamp01(max(r, 0.04)),
                                               clamp01(g), clamp01(max(b, 0.03)))

                // Roughness: soft-matte leaves; a touch smoother on the lit crown, rougher in
                // the shaded interior. Varies with leaf/clump → clears tell #3.
                ch.roughness[ch.idx(x, y)] = clamp01(0.66 - 0.08 * clump + 0.12 * leaf)

                // Height: leaf-layer micro-relief (bumpy canopy surface).
                ch.height[ch.idx(x, y)] = clamp01(0.40 + 0.30 * leaf + 0.14 * clump)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 1.4)
        return ch
    }

    /// Parameterizable unit paver surface: brick or concrete paver grid with per-unit
    /// color jitter, mortar joint, and edge wear.
    ///
    /// **Storage is the shared `UnitPatternParams` spine** — pavers and tile are the same object
    /// (units of a real size, laid in a bond, separated by a coloured joint), and this struct used
    /// to be that spine written a second time with different names: `bodyColor`/`mortarColor`/
    /// `unitWidth`/`unitHeight`/`colorJitter`/`seed` against tile's own set. One struct now; the
    /// paver-flavoured names below are forwarding views onto it, so paver code still reads as
    /// paver code and the two can no longer drift.
    ///
    /// Size is a REAL SIZE now, not a UV fraction. `unitWidth: 0.125` meant "1/8 of a bake tile",
    /// which is only a paver width if you also know the ground's uvScale — a number that could not
    /// be checked against reality. It is 0.25 m of brick, and says so.
    public struct PaverParams: Equatable, Hashable, Sendable, Codable {
        public var pattern: UnitPatternParams

        /// The ground mesh bakes its UVs at a fixed 2 m period, so the paver grid is NOT free to
        /// choose its own run the way an interior tile is — `isAdjustable: false` states that, and
        /// keeps the default grid at the 8-across it has always been.
        public static let groundTiling = SurfaceTiling(2.0, isAdjustable: false)

        public init(bodyColor: Vec3 = Vec3(0.58, 0.35, 0.24),
                    mortarColor: Vec3 = Vec3(0.52, 0.50, 0.47),
                    // 0.25 m × 0.125 m — a standard brick, and the same 8-across grid the old
                    // `unitWidth: 0.125` UV fraction produced at the ground's 2 m period.
                    unitSizeMeters: Double = 0.25,
                    // colorJitter: hard per-cell edges degrade seam score — keep ≤ 0.03.
                    // Smooth macro tone variation handles large-scale variation instead.
                    colorJitter: Double = 0.02, seed: UInt64 = 1) {
            pattern = UnitPatternParams(shape: .rectangle, unitSizeMeters: unitSizeMeters,
                                        bond: .stack, bodyColor: bodyColor,
                                        jointColor: mortarColor, jointFraction: 0.16,
                                        toneVariation: colorJitter, seed: seed)
        }

        public init(pattern: UnitPatternParams) { self.pattern = pattern }

        // Paver-flavoured views onto the shared spine.
        public var bodyColor: Vec3 {
            get { pattern.bodyColor }
            set { pattern.bodyColor = clampBand(newValue) }
        }
        public var mortarColor: Vec3 {
            get { pattern.jointColor }
            set { pattern.jointColor = clampBand(newValue) }
        }
        public var colorJitter: Double {
            get { pattern.toneVariation }
            set { pattern.toneVariation = clamp01(newValue) }
        }
        public var seed: UInt64 {
            get { pattern.seed }
            set { pattern.seed = newValue }
        }
        /// The grid this paver lays on the ground surface.
        public var layout: TileLayout { TileLayout.of(pattern, on: Self.groundTiling) }

        public static let redBrick = PaverParams(bodyColor: Vec3(0.58, 0.30, 0.22),
                                                  mortarColor: Vec3(0.74, 0.72, 0.68))
        public static let concretePaver = PaverParams(bodyColor: Vec3(0.70, 0.69, 0.67),
                                                       mortarColor: Vec3(0.62, 0.61, 0.59))
    }

    /// Unit paver grid — running bond by default; `PaverParams` controls size + color.
    public static func unitPavers(size: Int = MaterialGenerator.bakeSize, params: PaverParams = PaverParams()) -> MaterialChannels {
        // `.ground`, not `.tile`: unit pavers are an exterior patio/walkway surface (Phase 8
        // site system), so they must not appear in interior floor/wall/counter pickers.
        var ch = MaterialChannels(size: size, category: .ground)
        // Cell counts come from the shared layout — whole numbers by construction (UV period =
        // 1/N exact), which is all tile-safety ever required. (The old "must divide the bake"
        // rule was folklore; this generator's own 18 rows disproved it — see
        // `testEveryIntegerTileCountIsSeamless`.)
        let layout = params.layout
        let cellsU = layout.cols, cellsV = layout.rows
        // Floored at the thinnest drawable line, the same way tile is — this generator shares
        // `UnitPatternParams` AND the width control with tile, so it has to honour the same range
        // the control offers. It also has a hard-edged joint (no antialias band), which is exactly
        // the case a sub-texel line dissolves into a dotted mess.
        let jointWidth = layout.effectiveJointFraction(params.pattern.jointFraction) / 2
        for y in 0..<size {
            for x in 0..<size {
                // Texel-centre UVs prevent body/mortar mismatch at the wrap boundary.
                // u=x/size means x=size-1 → cu≈0.917 (body), but x=0 → cu=0 (mortar)
                // — a seam. With (x+0.5)/size, both boundary pixels land in mortar for
                // any cellsU where size > cellsU / (2 * jointWidth).
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                // Within-cell coords (integer cell count → wraps seamlessly at UV=1).
                let cu = (u * Double(cellsU)).truncatingRemainder(dividingBy: 1.0)
                let cv = (v * Double(cellsV)).truncatingRemainder(dividingBy: 1.0)
                // Mortar joint.
                let inMortar = cu < jointWidth || cu > 1.0 - jointWidth
                             || cv < jointWidth || cv > 1.0 - jointWidth
                // Macro noise for luma variation.
                let macro = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: params.seed ^ 0x42)
                // Per-unit color jitter from Voronoi cell ID (tile-safe).
                // Keep amplitude small so hard cell boundaries don't dominate seam score.
                let jitterCell = Noise.voronoiTiled(u, v, cells: cellsU, seed: params.seed)
                let jitter = (Double(jitterCell.cellId & 0xFFFF) / 65535.0 - 0.5) * params.colorJitter
                // Edge-wear (interior only).
                let wearU = cu < 0.5 ? cu : 1.0 - cu
                let wearV = cv < 0.5 ? cv : 1.0 - cv
                let edgeWear = clamp01(1.0 - (wearU + wearV) * 4.0)

                if inMortar {
                    // Mortar: body-color influenced by macro for tone variation.
                    let mLum = clamp01(params.mortarColor.x + 0.04 * macro)
                    ch.albedo[ch.idx(x, y)] = Vec3(mLum, mLum * (params.mortarColor.y / params.mortarColor.x),
                                                   mLum * (params.mortarColor.z / params.mortarColor.x))
                    ch.roughness[ch.idx(x, y)] = clamp01(0.78 + 0.08 * macro)
                    ch.height[ch.idx(x, y)] = 0.28
                } else {
                    let bx = params.bodyColor.x, by = params.bodyColor.y, bz = params.bodyColor.z
                    let r = clamp01(bx + jitter + 0.04 * macro)
                    let g = clamp01(by + jitter * (by / max(bx, 0.01)) + 0.04 * macro)
                    let b = clamp01(bz + jitter * (bz / max(bx, 0.01)) + 0.04 * macro)
                    ch.albedo[ch.idx(x, y)] = Vec3(r, g, b)
                    ch.roughness[ch.idx(x, y)] = clamp01(0.68 + 0.18 * edgeWear + 0.06 * macro)
                    ch.height[ch.idx(x, y)] = clamp01(0.62 - 0.18 * edgeWear + 0.06 * macro)
                }
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 2.5)
        // Fine masonry tooth on the brick/mortar faces — real pavers are gritty, not
        // molded plastic. Isotropic micro over both body and joint.
        addMicroDetail(&ch, seed: params.seed ^ 0xB8, baseCells: 90, strength: 0.60)
        return ch
    }

    /// Luxury vinyl plank (LVP) — embossed-in-register surface (grain follows the
    /// print layer below), plastic-satin finish, higher inter-board variation than
    /// real oak, plank seams more regular. `planks` must divide the tile evenly.
    public static func lvp(size: Int = MaterialGenerator.bakeSize, planks: Int = 4, seed: UInt64 = 77) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .wood)
        // S2.5 half 2 — planks along u, unbroken down v (see `woodGrain` for the shape).
        // Vinyl is a printed product, so its batch swing is smaller than solid wood's.
        ch.patternCells = Vec2(Double(max(1, planks)), 0)
        ch.patternJitter = planks > 1 ? 0.03 : 0
        // LVP grain tangent for subtle anisotropy (less pronounced than solid wood).
        var grain = [Vec2](repeating: Vec2(0, 1), count: size * size)
        let baseLight = Vec3(0.46, 0.36, 0.26)   // printed grain light band
        let baseDark  = Vec3(0.28, 0.20, 0.14)   // printed grain dark band
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let bf = u * Double(planks)
                let board = Int(floor(bf))
                let bx = bf - floor(bf)
                let bh = Noise.hash2(board, 0, seed)
                // Inter-board tone jitter (wider range than solid wood — mass production variety).
                let tone = 0.80 + 0.40 * Noise.unit(bh)
                // Plank seam at edges — hard line with thin bevel.
                let seamDist = min(bx, 1.0 - bx)
                let seam = smoothstep(0.0, 0.015, seamDist)   // 0 at seam edge → 1 inside plank
                // Printed grain: simple ring pattern (no domain warp — vinyl print is more
                // regular/mechanical than solid wood).
                let ringDensity = 6.0 + 4.0 * Noise.unit(bh ^ 0x33)
                let phase = Noise.unit(bh ^ 0x77) * 6
                let ring = bx * ringDensity + phase
                let lw = 0.5 - 0.5 * cos(2 * .pi * ring)
                // Subtle texture noise for emboss texture.
                let emboss = Noise.fbmTiled(u, v, baseCells: 8, octaves: 3, seed: seed ^ 0xAA)
                let col = mix(baseLight, baseDark, lw) * tone * (0.95 + 0.10 * emboss)
                ch.albedo[ch.idx(x, y)] = clampBand(col)
                // LVP is much shinier than solid wood (PU wear layer).
                ch.roughness[ch.idx(x, y)] = clamp01(0.30 + 0.10 * lw + 0.08 * (1 - seam))
                ch.height[ch.idx(x, y)] = clamp01(0.50 + 0.20 * lw + 0.20 * emboss - 0.15 * (1 - seam))
                grain[ch.idx(x, y)] = Vec2(0, 1)
            }
        }
        ch.grainTangent = grain
        ch.clearcoat = 0.22   // PU wear-layer gloss
        ch.deriveNormals(strength: 3)
        // Subtle embossed-in-register wood pore — premium LVP is textured, not glassy.
        // Kept light: the PU wear layer IS a satin plastic, so this is a whisper of tooth.
        addMicroDetail(&ch, seed: seed ^ 0xB9, baseCells: 110, strength: 0.35)
        return ch
    }

    /// **How much of a GLAZED or POLISHED surface's micro-relief may reach the occlusion
    /// channel**, as a fraction of its detail-normal strength.
    ///
    /// Same shape of argument as the metal case in `HouseRenderBridge.instanceData`, and the
    /// same lesson: the detail band's occlusion is a DIFFUSE self-shadowing term, so it is only
    /// physical where the micro-relief forms cavities that can trap light. A ceramic glaze is a
    /// vitreous film that has flowed and levelled — its "orange peel" is a smooth undulation,
    /// not a field of pits. It tilts the surface (which the detail NORMAL carries, and should)
    /// while occluding essentially nothing.
    ///
    /// Ignoring that is visible, not theoretical. With the S2.4 band first switched on at full
    /// strength, the `debugTester-materialsWalk` frame showed dark pepper scattered across the
    /// glazed subway tile, the square tile and the terrazzo chips: on a bright glossy white
    /// surface a 1 mm occlusion minimum does not read as tooth, it reads as **grime**. Those are
    /// the two materials Danny had just called good, so that is a regression, not a lift.
    ///
    /// Matte and porous surfaces — plaster, cork, leather, cloth, concrete, unglazed stone — keep
    /// the full authored strength, because their relief genuinely is cavities and the diffuse
    /// occlusion is the whole reason S2.4 exists.
    public static let glazeDetailOcclusionShare = 0.25

    /// Derive a tangent-space normal map from an arbitrary height array.
    /// Used for detail normals that are independent of the main `MaterialChannels.height`.
    /// Add a high-frequency micro-relief **detail normal** (pores / grit / weave) to a material
    /// that doesn't already author one. The engine samples `detailNormal` at `detailNormalUVScale ×
    /// uv` and blends it over the macro normal, so a 256² tile of fine noise gives close-range
    /// surface life on every wall/floor/counter — the cheap half of the "looks real" lift (audit
    /// worklist #1). No-op if the material already set its own detail normal.
    ///
    /// `grainAspect` stretches every feature by that factor — the anisotropic micro-relief a
    /// directional material needs — ALONG V by default (wood's pore channels run down the board;
    /// a round pit is what made oak read as corrugated), or ALONG U with `grainAlongU` (a brushed
    /// metal's hairline runs with its +U grain tangent). The cross-grain axis gets `aspect ×` the
    /// cells, so the coarsest octave keeps its along-grain count. It is a real `cellsX × cellsY`
    /// lattice (`Noise.fbmTiledAniso`): this used to multiply the coordinate instead, which tiles
    /// one strip `aspect` times — an exact repeat inside the tile (DH-0965).
    static func addMicroDetail(_ ch: inout MaterialChannels, seed: UInt64,
                               baseCells: Int = 96, octaves: Int = 2, strength: Double = 0.9,
                               occlusionStrength: Double? = nil, grainAspect: Int = 1,
                               grainAlongU: Bool = false) {
        guard ch.detailNormal == nil else { return }
        let n = ch.size
        let cross = baseCells * max(1, grainAspect)
        let cellsX = grainAlongU ? baseCells : cross
        let cellsY = grainAlongU ? cross : baseCells
        var h = [Double](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                let u = Double(x) / Double(n), v = Double(y) / Double(n)
                h[ch.idx(x, y)] = Noise.fbmTiledAniso(u, v, cellsX: cellsX, cellsY: cellsY,
                                                      octaves: octaves, seed: seed)
            }
        }
        setDetailRelief(&ch, height: h, strength: strength, occlusionStrength: occlusionStrength)
    }

    /// **The ONE place a detail band is written.** Takes the high-frequency height field and
    /// derives BOTH halves of it — the tangent-space `detailNormal` (specular-band micro-tilt)
    /// and S2.4's `detailOcclusion` (diffuse-visible micro-shadowing). One relief, one height
    /// field, one writer, so the two can never describe different surfaces
    /// ([[feedback-single-source-of-truth]]).
    ///
    /// `occlusionStrength` defaults to `strength` — the shared relief amount — and exists for
    /// the one case where those genuinely differ: a surface whose *normal* has to be held to a
    /// whisper for a reason that is specific to the normal. Linen is the worked example; its
    /// `strength: 0.03` is documented in-source as "a stronger detail normal sparkles into a
    /// dirty grain on a dimly-lit grazing panel", which is a statement about a specular-band
    /// artefact, not about how deep the weave is. The occlusion has no such failure mode (it
    /// multiplies the diffuse and mip-filters correctly), so inheriting that number would let
    /// a defect in one consumer silently suppress the other. Set it only with that kind of
    /// reason written down.
    static func setDetailRelief(_ ch: inout MaterialChannels, height: [Double], strength: Double,
                                occlusionStrength: Double? = nil) {
        ch.detailNormal = deriveNormalMap(height: height, size: ch.size, strength: strength)
        ch.detailOcclusion = deriveDetailOcclusion(height: height, size: ch.size,
                                                   strength: occlusionStrength ?? strength)
    }

    /// Micro-occlusion from a height field: how much of the hemisphere a texel loses to the
    /// micro-relief immediately around it.
    ///
    /// **The measure** is depth below the LOCAL surface level — `blur_R(h) − h` — not height
    /// against the tile mean. A pit is dark because its own rim occludes it; a low-lying but
    /// locally flat region is not occluded at all, and an absolute-height reading would wrongly
    /// darken it (and would double-print the macro tone the albedo already carries). The blur
    /// radius is one detail feature wide, so the term is band-limited to the detail frequency.
    ///
    /// **Amount** is driven by the same `strength` the normal uses — one relief authored once,
    /// read two ways — and clamped to `maxOcclusion` so no generator can crush a texel to black.
    /// The result is ≤ 1 everywhere with a mean a little under 1, which is exactly what makes it
    /// mip-safe: coarse levels converge to that mean instead of to "no effect".
    private static func deriveDetailOcclusion(height: [Double], size: Int, strength: Double) -> [Double] {
        let maxOcclusion = 0.5
        // One feature wide. `addMicroDetail`'s coarsest octave is ~60–130 cells across the
        // tile, so a feature is size/130 … size/60 texels; radius = size/128 sits at the fine
        // end of that and stays ≥ 1 for the 64² bakes the tests use.
        let r = max(1, size / 128)
        let n = size
        let wrap: (Int) -> Int = { i in ((i % n) + n) % n }
        // Separable box blur — toroidal, matching how the tile is sampled.
        var rowBlur = [Double](repeating: 0, count: n * n)
        let inv = 1.0 / Double(2 * r + 1)
        for y in 0..<n {
            for x in 0..<n {
                var s = 0.0
                for d in -r...r { s += height[y * n + wrap(x + d)] }
                rowBlur[y * n + x] = s * inv
            }
        }
        var depth = [Double](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                var s = 0.0
                for d in -r...r { s += rowBlur[wrap(y + d) * n + x] }
                depth[y * n + x] = Swift.max(0, s * inv - height[y * n + x])
            }
        }
        // Normalise by the field's OWN depth RMS, then scale to a target contrast.
        //
        // This is deliberate and it is what the detail NORMAL already does implicitly:
        // `deriveNormalMap` normalises every gradient to a unit vector, so `strength` sets
        // the micro-TILT and the height field's absolute amplitude drops out. Reading raw
        // height differences here put the occlusion on a completely different footing — the
        // library's baked detail fields differ 16× in depth RMS, so a fixed gain gave linen
        // σ 0.004 (invisible) and asphalt σ 0.093 (dirt). Measured on the real Metal path,
        // the added micro-contrast tracks σ almost exactly, so an un-normalised bake means
        // "does this material show micro-relief?" is decided by an incidental authoring
        // amplitude rather than by the surface.
        //
        // `strength` still modulates — a generator that asks for gentle relief gets gentler
        // occlusion — but within a band, so no material lands at invisible or at grime.
        // …but a strength of ZERO means OFF, not "the floor". The band below floors at 0.30 so
        // that a generator asking for gentle relief still gets *visible* relief — but that
        // floor made `occlusionStrength: 0` yield contrast 0.048 instead of nothing, so a
        // material could not opt out at all. That is a real trap: it silently gave interior
        // PAINT a micro-occlusion and regressed the wall-splotch blotch 0.07 → 0.25, undoing
        // a third of the SSAO hemisphere fix. Every other opt-in term in this engine treats 0
        // as an exact no-op; so does this one now.
        guard strength > 0 else { return [Double](repeating: 1, count: n * n) }
        let rms = (depth.reduce(0) { $0 + $1 * $1 } / Double(n * n)).squareRoot()
        let contrast = 0.16 * Swift.min(1.4, Swift.max(0.30, max(0, strength) / 0.7))
        let gain = rms > 1e-9 ? contrast / rms : 0
        var occ = [Double](repeating: 1, count: n * n)
        for i in 0..<(n * n) {
            occ[i] = 1 - Swift.min(maxOcclusion, depth[i] * gain)
        }
        // MEAN-NEUTRALISE. Micro-occlusion must modulate a surface AROUND its value, not
        // lower it: the material's overall lightness is the author's decision (and, for a
        // metal, F0 itself), while this band's job is purely the structure. Un-normalised, the
        // field's mean sits below 1 and every material silently darkens — measured, that
        // pushed matte-black's specular IBL *below its single-scatter level* (0.01043 vs the
        // 0.010908 → 0.012121 the multi-scatter fix had just won, i.e. it read as if S1.3b had
        // been reverted) and shifted a whole test frame by 20 levels. Both were real gate
        // failures, and both were this. Re-centring costs nothing: the spatial structure —
        // which is the entire point — is untouched, only the DC term is.
        let occMean = occ.reduce(0, +) / Double(n * n)
        if occMean > 1e-6 {
            for i in 0..<(n * n) { occ[i] = Swift.min(1.0, occ[i] / occMean) }
        }
        return occ
    }

    private static func deriveNormalMap(height: [Double], size: Int, strength: Double) -> [Vec3] {
        var normals = [Vec3](repeating: Vec3(0, 0, 1), count: size * size)
        let wrap: (Int) -> Int = { i in ((i % size) + size) % size }
        for y in 0..<size {
            for x in 0..<size {
                let hL = height[wrap(y) * size + wrap(x - 1)]
                let hR = height[wrap(y) * size + wrap(x + 1)]
                let hD = height[wrap(y - 1) * size + wrap(x)]
                let hU = height[wrap(y + 1) * size + wrap(x)]
                normals[y * size + x] = normalize3(Vec3((hL - hR) * strength, (hD - hU) * strength, 1))
            }
        }
        return normals
    }

    // MARK: - Framed art print (a PICTURE, not a tiling PBR surface)

    /// The default artwork inside a wall / tabletop picture frame — a soft, muted horizon
    /// landscape. This is a PICTURE mapped ONCE across the frame's art panel (the art plate carries
    /// authored 0…1 UVs which `HouseSceneBuilder.convertMeshUV` honours, and the render instance
    /// forces the hex anti-tiling de-repeat OFF), NOT a tiling material. It is therefore
    /// deliberately **absent from `MaterialRegistry.all`** and **not swept by `TextureAudit`**:
    /// TextureAudit checks tiling-surface tells (edge-wrap seam score, flat-roughness, grain
    /// coherence, macro contrast) that are meaningless for a one-shot representational image — a
    /// landscape *should* have a sky-to-ground gradient (a "seam" top-to-bottom) and a near-flat
    /// print roughness. Auditing it as a tile would wrongly fail it; instead it is resolved directly
    /// on the art sub-panel in `HouseRenderBridge+Frame`.
    ///
    /// A calm, low-saturation composition: a dusty-blue sky warming to a cream haze at the horizon,
    /// a soft off-centre sun glow, and three layered rolling hills darkening toward a muted-sage
    /// foreground. Deterministic (seeded — the seed varies the ridgelines). Matte (no
    /// clearcoat/glass glare) so the picture reads clearly under room light.
    ///
    /// **Single lever (Danny):** to swap the default art, edit this palette (or point
    /// `FrameSpec.softMatteArt` at a different assignment). A real photo import / art picker remains
    /// the flagged taste follow-up.
    public static func landscapePrint(size: Int = MaterialGenerator.bakeSize,
                                      seed: UInt64 = 0x5A11) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .paint)   // dielectric; de-repeat forced off per-instance
        // Linear-RGB palette (muted / low-saturation dusk).
        let skyTop   = Vec3(0.30, 0.42, 0.58)   // dusty blue (zenith)
        let skyHaze  = Vec3(0.74, 0.66, 0.52)   // warm cream at the horizon
        let sun      = Vec3(0.96, 0.90, 0.78)   // soft warm glow
        let hillFar  = Vec3(0.30, 0.36, 0.40)   // hazy blue-grey ridge
        let hillMid  = Vec3(0.22, 0.28, 0.24)   // sage
        let hillNear = Vec3(0.14, 0.19, 0.13)   // deeper olive foreground
        let horizonV = 0.63                      // horizon a touch below centre (sky-dominant, calm)
        let sunU = 0.34, sunV = 0.30             // soft sun position (upper-left)

        func lerp(_ a: Vec3, _ b: Vec3, _ t: Double) -> Vec3 { a + (b - a) * clamp01(t) }
        func ss(_ t: Double) -> Double { let x = clamp01(t); return x * x * (3 - 2 * x) }
        // Ridgeline: the v of the ridge crest at column u — seeded, horizontally seamless (tiled).
        func ridge(baseV: Double, amp: Double, cells: Int, s: UInt64) -> (Double) -> Double {
            { u in baseV - amp * (Noise.fbmTiled(u, 0.5, baseCells: cells, octaves: 3, seed: s) - 0.5) }
        }
        let rFar  = ridge(baseV: horizonV + 0.02, amp: 0.05, cells: 5, s: seed ^ 0x11)
        let rMid  = ridge(baseV: horizonV + 0.12, amp: 0.09, cells: 4, s: seed ^ 0x22)
        let rNear = ridge(baseV: horizonV + 0.26, amp: 0.13, cells: 3, s: seed ^ 0x33)

        for y in 0..<size {
            let v = (Double(y) + 0.5) / Double(size)     // 0 = top (sky), 1 = bottom (foreground)
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                // Sky: vertical blend zenith → horizon haze, plus a soft radial sun glow.
                var c = lerp(skyTop, skyHaze, ss(v / horizonV))
                let du = u - sunU, dv = v - sunV
                let sunDist = (du * du + dv * dv).squareRoot()
                let glow = clamp01(1.0 - sunDist / 0.42)
                c = lerp(c, sun, 0.55 * glow * glow)
                // Hills — painted back-to-front so nearer, darker ridges overwrite.
                if v >= rFar(u)  { c = lerp(c, hillFar, 0.85) }
                if v >= rMid(u)  { c = hillMid }
                if v >= rNear(u) { c = hillNear }
                // A whisper of tonal break so it isn't a dead-flat print (drives roughness too).
                let grain = Noise.fbmTiled(u, v, baseCells: 40, octaves: 2, seed: seed ^ 0x9E) - 0.5
                c = c * (1.0 + 0.05 * grain)
                ch.albedo[ch.idx(x, y)] = clampBand(c)
                ch.roughness[ch.idx(x, y)] = clamp01(0.66 + 0.06 * grain)
                ch.height[ch.idx(x, y)] = 0.5
            }
        }
        ch.deriveNormals(strength: 0)   // a flat print — no surface relief
        return ch
    }

    // MARK: – Glazing (§ window glass)

    /// How many metres of real wall one glazing tile spans. Ten reeds across the tile, so
    /// `FlutedGlassProfile.pitch` divides it exactly and the bake is seamless (toroidal).
    public static let glazingTileMeters = FlutedGlassProfile.pitch * 10

    /// Optically clear float glass. A pane's *appearance* comes from the engine's glass
    /// pass (Fresnel + refraction), not from a sampled texture — these channels exist so
    /// the glazing picker can show a real baked swatch/preview sphere for the material
    /// the user is choosing, and so the auditors can sweep glazing like any other family.
    public static func clearGlass(size: Int = MaterialGenerator.bakeSize,
                                  seed: UInt64 = 311) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .glass)
        let base = Vec3(0.80, 0.86, 0.88)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size), v = (Double(y) + 0.5) / Double(size)
                // Float glass is drawn on a tin bath: a very faint long-wave undulation is
                // the only real surface structure it has. Enough to keep roughness from
                // being dead-flat (the TextureAudit tell) without frosting it.
                // Amplitudes follow the `chrome` precedent: enough spatial variation to
                // clear TextureAudit's flat-roughness / flat-colour tells while the surface
                // still reads optically polished. The pane's REAL roughness is
                // `GlassMaterial.optics.roughness` (0.02) — this map only feeds the swatch.
                let wave = Noise.fbmTiled(u, v, baseCells: 4, octaves: 4, seed: seed)
                ch.albedo[ch.idx(x, y)] = clampBand(base * (0.94 + 0.12 * wave))
                ch.roughness[ch.idx(x, y)] = clamp01(0.04 + 0.16 * wave)
                ch.height[ch.idx(x, y)] = clamp01(0.5 + 0.04 * (wave - 0.5))
            }
        }
        ch.clearcoat = 0.6                 // polished dielectric surface
        ch.deriveNormals(strength: 1.0)
        return ch
    }

    /// Body-tinted BRONZE glass — the surface is the same polished float glass; the colour
    /// comes from absorption through the body (`GlassMaterial.Optics.density`), so the
    /// albedo carries the tint and the roughness stays clear-glass low.
    public static func bronzeGlass(size: Int = MaterialGenerator.bakeSize,
                                   seed: UInt64 = 313) -> MaterialChannels {
        var ch = clearGlass(size: size, seed: seed)
        let tint = Vec3(0.62, 0.48, 0.33)
        for i in ch.albedo.indices { ch.albedo[i] = clampBand(ch.albedo[i] * tint * 1.35) }
        return ch
    }

    /// FLUTED / reeded obscuring glass. Every channel is derived from
    /// `FlutedGlassProfile` — the same profile `HouseMesh.glazing` extrudes into real
    /// vertical reeds on the pane — so the swatch and the rendered pane describe one
    /// object. Nothing here raises roughness: the obscuring is shape, not scatter (a rough
    /// dielectric is a stochastic ray estimator, i.e. the "live noise" this replaces).
    public static func flutedGlass(size: Int = MaterialGenerator.bakeSize,
                                   seed: UInt64 = 317) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .glass)
        let base = Vec3(0.80, 0.85, 0.86)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size), v = (Double(y) + 0.5) / Double(size)
                let s = u * glazingTileMeters                       // metres along the pane
                let lobe = FlutedGlassProfile.relief(at: s) / FlutedGlassProfile.depth  // 0…1
                let wave = Noise.fbmTiled(u, v, baseCells: 4, octaves: 4, seed: seed)
                // Albedo stays the clear-glass body (same float-glass variation) — the reeds
                // read through SHADING (normals), not painted-in stripes, or the swatch would
                // promise a pattern the render doesn't have.
                ch.albedo[ch.idx(x, y)] = clampBand(base * (0.94 + 0.12 * wave))
                // A whisper more roughness in the seams between reeds (mould witness line).
                ch.roughness[ch.idx(x, y)] = clamp01(0.04 + 0.14 * wave + 0.04 * (1 - lobe))
                ch.height[ch.idx(x, y)] = clamp01(lobe)
            }
        }
        ch.clearcoat = 0.6
        // Strong enough that the half-round cross-section reads at swatch size; the pane
        // itself gets the true shape from the extruded footprint, not from this map.
        ch.deriveNormals(strength: 6.0)
        return ch
    }

    // MARK: – Glass block (both the wall finish AND the window glazing)

    // A glass block is a hollow moulded UNIT laid up in mortar, like brick — and it is also
    // glass, so it belongs in BOTH pickers. It ships as two registry entries because a
    // material category answers two different questions in two different places, and a glass
    // block genuinely answers them differently depending on where it is laid:
    //
    //   • `glass-block` (`.brick`) — the WALL finish. `.brick` is right on both axes the
    //     category decides: APPLICABILITY (`MaterialApplicability` → wall + backsplash, which
    //     is where a glass-block partition goes) and ANTI-TILING (`ElementMaterial
    //     .coherentPatternCategories` already lists `.brick`, so the grid opts OUT of the
    //     hex-stochastic de-repeat — a coherent grid double-prints its joints under the blend).
    //   • `glass-block-glazing` (`.glass`) — blocks laid up INSIDE a window opening. `.glass`
    //     routes the pane into the engine's dielectric pass, which is the only way anything in
    //     this app transmits light. That path never samples an albedo slice, so the block grid
    //     cannot come from this bake: it comes from real pane GEOMETRY
    //     (`HouseMesh.glassBlockPane`), exactly as fluted glass's reeds do.
    //
    // Both bakes and the pane geometry read ONE description of the block — `GlassBlockProfile`
    // — so the wall, the window and the two swatches can never disagree about the module.

    /// Nominal glass-block module — see `GlassBlockProfile`, the single source. These forward
    /// so existing call sites keep reading the same one fact.
    public static var glassBlockFaceMeters: Double     { GlassBlockProfile.faceMeters }
    public static var glassBlockMortarMeters: Double   { GlassBlockProfile.mortarMeters }
    public static var glassBlockModuleMeters: Double   { GlassBlockProfile.moduleMeters }
    public static var glassBlockWallTileMeters: Double { GlassBlockProfile.wallTileMeters }
    public static var glassBlockPerTile: Int           { GlassBlockProfile.perTile }
    public static var glassBlockTileMeters: Double     { GlassBlockProfile.tileMeters }

    /// GLASS BLOCK — a laid-up grid of thick moulded blocks with recessed mortar joints, each
    /// block's face carrying the moulded WAVE flutes that make a glass block obscuring.
    ///
    /// Three layers, each doing one job:
    ///  1. the **joint** — a recessed, matte, cementitious line on the 200 mm grid (the coherent
    ///     pattern, and the reason the material opts out of the hex de-repeat);
    ///  2. the **pillow** — a block face is proud in the middle and chamfered back to the joint,
    ///     which is what catches the highlight along every course;
    ///  3. the **flutes** — a per-block wave, its direction alternating by a per-block hash so the
    ///     wall reads as individually laid units instead of one printed sheet. The flute
    ///     coordinate is block-LOCAL and completes a whole number of periods per block, so it is
    ///     continuous at every joint and the tile stays toroidal.
    ///
    /// Roughness is deliberately three-valued in space — polished flute crests, mould-textured
    /// troughs, matte mortar — so it clears TextureAudit's flat-roughness tell honestly rather
    /// than by dithering a constant.
    public static func glassBlock(size: Int = MaterialGenerator.bakeSize,
                                  blocks: Int = MaterialGenerator.glassBlockPerTile,
                                  seed: UInt64 = 331) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .brick)
        let n = max(1, blocks)
        // S2.5 half 2 — a stack-bond n×n block grid: exact cell counts on both axes.
        // Glass casting tone varies block to block only subtly.
        if n > 1 { ch.patternCells = Vec2(Double(n), Double(n)); ch.patternJitter = 0.03 }
        // Joint half-width in block-module units, straight off the real 10 mm-in-200 mm module.
        let jointHalf = GlassBlockProfile.jointHalfFraction
        let glassBody = Vec3(0.66, 0.73, 0.74)     // pale aqua cast of thick soda-lime glass
        let mortarBody = Vec3(0.56, 0.55, 0.52)    // grey cementitious joint
        let flutesPerBlock = GlassBlockProfile.flutesPerBlock   // moulded waves across one face

        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let bu = u * Double(n), bv = v * Double(n)
                let bi = Int(floor(bu)), bj = Int(floor(bv))
                let fx = bu - floor(bu), fy = bv - floor(bv)
                let edge = min(min(fx, 1 - fx), min(fy, 1 - fy))       // 0 at a joint centre
                let joint = 1 - smoothstep(jointHalf, jointHalf + 0.020, edge)
                let pillow = smoothstep(jointHalf, 0.20, edge)         // 0 at the joint … 1 mid-face

                // Per-block identity: flute direction + a slight casting tone, so no two
                // neighbours are clones. Both read from ONE hash of the block index.
                let h = Noise.hash2(bi, bj, seed)
                let along = (h & 1) == 0 ? fy : fx
                // Whole periods per block ⇒ value + derivative match at both block edges.
                let flute = 0.5 - 0.5 * cos(2 * Double.pi * flutesPerBlock * along)
                let tone = 0.90 + 0.20 * Noise.unit(Noise.hash2(bi, bj, seed ^ 0x9E37))

                // The block's hollow cavity scatters transmitted light into a soft mottle —
                // this is what "obscuring" looks like from outside, and it supplies macro contrast.
                let cavity = Noise.fbmTiled(u, v, baseCells: n * 3, octaves: 4, seed: seed ^ 0x5A)
                let face = glassBody * tone * (0.84 + 0.30 * flute) * (0.94 + 0.12 * cavity)
                let mortarNoise = Noise.fbmTiled(u, v, baseCells: n * 10, octaves: 3, seed: seed ^ 0x2B)
                let mortarCol = mortarBody * (0.90 + 0.20 * mortarNoise)
                ch.albedo[ch.idx(x, y)] = clampBand(mix(face, mortarCol, joint))

                // Moulded glass is glossy but not optically flat: crests polish, troughs keep the
                // mould's tooth, the chamfer near the joint is scuffed, and the mortar is matte.
                let micro = (Noise.fbmTiled(u, v, baseCells: 26, octaves: 3, seed: seed ^ 0x77) - 0.5) * 0.06
                let glassRough = 0.10 + 0.18 * (1 - flute) + 0.07 * (1 - pillow) + micro
                ch.roughness[ch.idx(x, y)] = clamp01(mix(glassRough, 0.86 + 0.10 * mortarNoise, joint))

                // Relief: the unit stands proud of its joint, the flutes ride on the face.
                let faceH = 0.55 + 0.28 * pillow + 0.15 * flute * pillow
                ch.height[ch.idx(x, y)] = clamp01(mix(faceH, 0.08, joint))
            }
        }
        ch.clearcoat = 0.55                    // moulded glass keeps a real specular top lobe
        ch.deriveNormals(strength: 6)
        // The joint's sand tooth + the block's mould texture — without it the face reads plastic.
        addMicroDetail(&ch, seed: seed ^ 0xD3, baseCells: 90, strength: 0.35)
        return ch
    }

    /// GLASS BLOCK **as glazing** — the swatch for `GlassMaterial.block`, the window pane made
    /// of laid-up blocks that really transmits light.
    ///
    /// This bake is a SWATCH ONLY. A glazing draw group is routed into the engine's dielectric
    /// pass and never gets an albedo/roughness slice (see `HouseSceneBuilder.glass(for:)`), so
    /// what the user sees in the window is `HouseMesh.glassBlockPane` — real pillowed blocks in
    /// real mortar valleys, from the SAME `GlassBlockProfile` this reads. The picker chip and
    /// the pane therefore describe one object, and this map can never promise a pattern the
    /// render doesn't have.
    ///
    /// It differs from the wall bake in exactly the way the real thing differs: seen as
    /// glazing, a block is lit THROUGH — pale and luminous rather than a grey-jointed masonry
    /// face — and every surface is optically smooth (the obscuring is the moulded shape, not
    /// scatter; a rough dielectric would make the RT pass a per-frame re-seeded estimator).
    public static func glassBlockGlazing(size: Int = MaterialGenerator.bakeSize,
                                         blocks: Int = GlassBlockProfile.perTile,
                                         seed: UInt64 = 337) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .glass)
        let n = max(1, blocks)
        let body = Vec3(0.80, 0.87, 0.87)          // backlit soda-lime glass, faint aqua
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let bu = u * Double(n), bv = v * Double(n)
                let bi = Int(floor(bu)), bj = Int(floor(bv))
                let fx = bu - floor(bu), fy = bv - floor(bv)
                // The ONE relief description — the same function the pane's grid is displaced
                // by, sampled here per texel. 0 in the joint valley, 1 at the crown.
                let lobe = GlassBlockProfile.relief(fx, fy)
                    * GlassBlockProfile.amplitudeScale(block: bi, bj, seed: seed)
                // The hollow cavity scatters transmitted light into a soft mottle — this is
                // what "obscuring" looks like when you light a block from behind.
                let cavity = Noise.fbmTiled(u, v, baseCells: n * 3, octaves: 4, seed: seed ^ 0x5A)
                let tone = 0.94 + 0.12 * Noise.unit(Noise.hash2(bi, bj, seed ^ 0x9E37))
                // Brighter at the crown (more glass lit through), dimmer down in the joint.
                ch.albedo[ch.idx(x, y)] = clampBand(body * tone
                                                    * (0.72 + 0.34 * lobe)
                                                    * (0.94 + 0.12 * cavity))
                // Optically smooth everywhere — a whisper more in the mould-textured valley.
                ch.roughness[ch.idx(x, y)] = clamp01(0.03 + 0.06 * (1 - lobe) + 0.12 * cavity)
                ch.height[ch.idx(x, y)] = clamp01(lobe)
            }
        }
        ch.clearcoat = 0.6                         // polished dielectric surface
        // Strong enough that the pillowed grid reads at swatch size; the pane itself gets the
        // true shape from its displaced geometry, not from this map.
        ch.deriveNormals(strength: 6)
        return ch
    }
}
