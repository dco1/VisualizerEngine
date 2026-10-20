import simd
import Foundation

extension MaterialGenerator {
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
}
