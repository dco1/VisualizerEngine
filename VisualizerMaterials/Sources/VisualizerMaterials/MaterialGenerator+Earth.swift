import simd
import Foundation

extension MaterialGenerator {
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
}
