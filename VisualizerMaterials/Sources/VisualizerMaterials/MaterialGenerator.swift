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
