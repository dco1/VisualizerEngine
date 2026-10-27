import simd
import Foundation

extension MaterialGenerator {
    /// Velvet — the **sheen** lobe (§3) made the whole point: a *dark* base with a
    /// strong retroreflective grazing sheen and a fine woven nap normal. Sheen, not
    /// roughness, carries the look, so the channels stay matte and dark while the
    /// `sheen` scalar drives the engine's grazing lobe.
    public static func velvet(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 9,
                              base: Vec3 = Vec3(0.16, 0.07, 0.22)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .fabric)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // fine weave: two high-frequency bands cross to make the nap
                // DH-1339: was two crossed 48-cycle bands (a basket weave — velvet has NO visible weave). Now an isotropic fine
                // nap field of the same frequency, so the base carries pile tone, not a lattice.
                let nap = 0.5 * Noise.fbmTiled(u * 48, v * 48, baseCells: 4, octaves: 2, seed: seed)
                        + 0.5 * Noise.fbmTiled(u * 31, v * 37, baseCells: 4, octaves: 2, seed: seed ^ 0x5E)
                let macro = 0.82 + 0.36 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed ^ 0x6F)
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * (0.85 + 0.3 * nap))
                ch.roughness[ch.idx(x, y)] = clamp01(0.84 + (nap - 0.5) * 0.16)   // matte cloth, nap-broken
                ch.height[ch.idx(x, y)] = nap
            }
        }
        ch.sheen = 0.85                            // velvet is sheen-dominated
        ch.diffuseRoughness = 1.0                  // DH-1301 — OpenPBR's velvet example: 1.0
        ch.sheenRoughness = 0.45                    // dense cut pile → a BROAD, soft grazing glow
                                                    // (DH-0081; was the 0.30 library default)
        ch.deriveNormals(strength: 0.7)      // was 2 on the crossed-band weave; on the isotropic nap that read as crushed suede
        // DH-1316: the NAP — the pile leans one way (the fibre angle below, 0.6 rad), swirled and crushed in ~5 cm
        // patches: ~6° of lean, invisible face-on, the light/dark pile marks at grazing (MaterialGenerator+FabricLook).
        if refinedFabricLook {
            applyNapLean(&ch, seed: seed ^ 0x4A9, angle: 0.6, lean: 0.11, patchCells: 6, swirl: 1.1, crushLift: 0.07)
        }
        // Very fine sub-thread nap so the pile catches grazing light as fuzz, not plastic.
        // DH-1339: real pile fibre. ~1 mm cut fibres (14 texels at ~0.07 mm) ~0.13 mm wide, laid mostly one way with a wide
        // scatter — the nap — added (not layered) so they pile into a soft fuzz rather than read as hairs.
        addFibreDetail(&ch, seed: seed ^ 0xB3, fibres: [
            .init(count: 26000, length: 14, width: 1.8, angle: 0.6, spread: 1.1, layered: false, widthJitter: 0.3)],
            baseCells: 120, strength: 0.7, occlusionStrength: 0.4)
        return ch
    }

    /// Wool / bouclé — looped-yarn upholstery texture. Coarser than velvet, with chunky
    /// loop pile visible as raised bumps. Medium sheen at grazing only.
    public static func wool(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 137,
                            base: Vec3 = Vec3(0.72, 0.68, 0.64)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .fabric)
        let sh = seed
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                // Chunky loop: medium-frequency Voronoi cells are the "loops"
                let loop = Noise.voronoiTiled(u, v, cells: 16, jitter: 0.9, seed: sh)
                let loopHeight = smoothstep(0.0, 0.25, loop.f1) * 0.8 + 0.2
                // Fine fiber within each loop
                let fiber = Noise.fbmTiled(u * 16, v * 16, baseCells: 4, octaves: 3, seed: sh ^ 0x3C)
                let macro = 0.80 + 0.40 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh ^ 0x5F)
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * (0.70 + 0.50 * loopHeight * (0.6 + 0.4 * fiber)))
                // Loops catch more light at top → rougher where fibers cross at base
                ch.roughness[ch.idx(x, y)] = clamp01(0.74 + (1 - loopHeight) * 0.18
                    + (fiber - 0.5) * 0.10)
                ch.height[ch.idx(x, y)] = clamp01(loopHeight * 0.8 + fiber * 0.08)
            }
        }
        // RESTORED to 0.45 (Danny, 2026-08-22). `a046075` — a LINEN-only refinement — took this
        // to 0.0 as collateral, and the tell it was collateral is four lines up: the docstring
        // has said "Medium sheen at grazing only" throughout. Wool is what every rug wears and
        // is a user-pickable finish, and a correct sheen lobe in `brdf()` cannot rescue a
        // material that asks for none. Sits between linen (0.30) and velvet (0.85), which is
        // what "medium" means here. `MaterialTextureTests.testEveryUncoatedFabricCarriesSheen`
        // now iterates the registry, so no future fabric can lose it silently.
        ch.sheen = 0.45
        ch.diffuseRoughness = 0.80                  // DH-1301 — a looped bouclé: rough, a shade short of cut pile
        ch.sheenRoughness = 0.60                    // chunky looped bouclé → the broadest, fuzziest
                                                    // nap of the fabric set (DH-0081)
        ch.deriveNormals(strength: 4)
        // Soft fibrous fuzz over the chunky loops — the between-loop wool haze at close range.
        // DH-1339: the fuzzy halo of real wool — 3–5 mm loose fibres (~0.2 mm wide) in every direction over the loops.
        addFibreDetail(&ch, seed: seed ^ 0xB4, fibres: [
            .init(count: 5500, length: 60, width: 3.0, angle: 0, spread: .pi, layered: false, widthJitter: 0.5),
            .init(count: 600, length: 110, width: 4.5, angle: 0, spread: .pi, layered: true, widthJitter: 0.4)],
            baseCells: 60, strength: 1.0, occlusionStrength: 0.45)   // retuned after the 2 cm A/B preferred the old wool
        return ch
    }

    /// Cork flooring — pressed bark agglomerate, laid as tiles.
    ///
    /// **What the previous bake got wrong** (Danny: "looks like cells under a
    /// microscope"). Every dark feature was driven off the Voronoi *boundary distance*
    /// `f2 - f1`, at two frequencies. That paints a connected web along every cell edge
    /// and leaves the interiors dead flat — a cell-wall generator, i.e. literally a
    /// diagram of plant cells. Real cork has NO boundary web. Its structure is
    /// (a) filled granule chips whose **interiors** differ in tone from one another, and
    /// (b) discrete dark **pores** scattered inside them. Both were missing.
    ///
    /// **The cues that make it read as cork flooring**, in the order the eye takes them.
    /// All are sized against this bake's budget: the floor's UV tile is 2.0 m over 512
    /// texels = **3.9 mm/texel**, so nothing finer than ~12 mm can be drawn cleanly and
    /// the sub-texel tooth is left to the detail normal.
    ///
    /// 1. **Tile layout** — 6 × 6 over the 2.0 m UV tile → 33 cm squares, quarter-turned
    ///    like parquet. Cork is a *laid floor*, and the tonal jump from tile to tile is
    ///    the single strongest "this is flooring, not a photo of bark" cue. The lattice
    ///    is offset a half tile so the bake's own wrap falls MID-tile: a joint sitting on
    ///    the wrap would read as a repeat seam (and `TextureAudit` would flag it).
    /// 2. **Chips, not cells** — each granule takes its tone from its `cellId`, a lottery
    ///    over cream / honey / russet. The boundary gets a hairline and nothing more.
    /// 3. **Lenticels** — cork's signature, and the thing whose absence gave the game
    ///    away: the pore channels through the bark read as near-black dashes ≈ 12 × 5 mm,
    ///    elongated along their chip's axis, in DRIFTS — porosity is drawn per chip, so
    ///    some flakes are peppered and some nearly clean.
    /// 4. **Flattened, directional flakes** — granules are pressed, so they're elongated
    ///    ~2:1 (`voronoiTiledAniso`), and each floor tile reads a rotated, offset patch of
    ///    the field so no flake ever runs across a joint.
    /// 5. **Fibrous striation** along a chip's long axis — bark's radial fibre sliced open,
    ///    the "combed" texture inside every flake.
    /// 6. **Sparse burl flecks** — a few much darker, irregular bark clumps per tile.
    /// 7. **Warm honey palette** — amber through russet, replacing a desaturated grey-tan.
    /// 8. **Factory satin seal** — an EVEN clearcoat, with roughness rising only where the
    ///    seal thins (in the pores, along the joints), not as random fine noise.
    ///
    /// `grainTangent` stays a single uniform axis deliberately. `TextureAudit` requires
    /// grain coherence ≥ 0.5 for `.wood`, and a quarter-turn tile field averages to
    /// EXACTLY 0 — θ and θ+90° cancel in the axial order parameter. The seal is also
    /// rolled across the whole floor in one pass, so one axis is the honest answer; the
    /// chip direction lives in albedo/height, where the quarter turn can't cancel it.
    public static func cork(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 103) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .wood)
        let sh = seed

        // Per-granule tone lottery — the spread the reference boards show between
        // neighbouring flakes. Chips are FILLED with these; nothing is painted along an edge.
        // These are LINEAR albedo and are deliberately darker and far more saturated than a
        // first guess suggests: the reference boards mean about sRGB (200, 165, 118), i.e.
        // linear ≈ (0.58, 0.37, 0.18) with a red:blue ratio over 3. A palette in the 0.75+
        // linear range with ratio ~2 (what this generator used to carry) bakes out as
        // oatmeal/limestone — the right *pattern* in the wrong material.
        let chipTones = [Vec3(0.76, 0.55, 0.30),   // pale cream flake
                         Vec3(0.68, 0.47, 0.24),
                         Vec3(0.60, 0.39, 0.19),   // honey body
                         Vec3(0.52, 0.32, 0.15),
                         Vec3(0.42, 0.24, 0.11)]   // russet flake
        let poreColor = Vec3(0.16, 0.09, 0.05)     // lenticel channel — near-black brown
        let burlColor = Vec3(0.26, 0.15, 0.08)     // dark bark clump
        let toneCount = UInt64(chipTones.count)

        let tiles = 6                 // over the floor's 2.0 m UV tile → 33 cm cork tiles
        // Chip lattice in cells across the whole bake. 34 × 76 over 2.0 m → flakes
        // ≈ 59 × 26 mm (15 × 7 texels at 512): the coarse-flake cork of the reference,
        // and about the smallest chip that still resolves at 3.9 mm/texel.
        let chipX = 34, chipY = 76
        let fineX = 70, fineY = 150   // sub-granule layer — breaks the big flakes up
        let poreCells = 150           // ≈ 13 mm lattice; only some cells hold a pore

        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)

                // ── 1. Tile lattice, offset a half tile so the bake's wrap lands mid-tile.
                let tfu = u * Double(tiles) + 0.5, tfv = v * Double(tiles) + 0.5
                let ti = Noise.wrap(Int(floor(tfu)), tiles), tj = Noise.wrap(Int(floor(tfv)), tiles)
                let lu = tfu - floor(tfu), lv = tfv - floor(tfv)     // 0…1 within the tile
                let th = Noise.hash2(ti, tj, sh ^ 0xC0DE)
                let turned = Noise.unit(th) < 0.5                    // quarter-turn parquet
                // Each tile reads its own patch of the chip field (the lattice is periodic
                // with period 1, so an arbitrary offset stays seamless at the bake wrap).
                let gu = (turned ? v : u) + Noise.unit(Noise.mix(th))
                let gv = (turned ? u : v) + Noise.unit(Noise.mix(th ^ 0x9E37))

                // ── 2. Chips coloured by cellId — the interior lottery, NOT f2 - f1.
                let chip = Noise.voronoiTiledAniso(gu, gv, cellsX: chipX, cellsY: chipY,
                                                   jitter: 0.95, seed: sh ^ 0x11)
                var col = chipTones[Int(chip.cellId % toneCount)]
                let fine = Noise.voronoiTiledAniso(gu, gv, cellsX: fineX, cellsY: fineY,
                                                   jitter: 0.95, seed: sh ^ 0x5A)
                col = mix(col, chipTones[Int(fine.cellId % toneCount)], 0.30)
                // …plus a continuous per-chip brightness draw on top of the 5-tone lottery, so
                // no two neighbouring flakes match exactly. Without this the chip mosaic is
                // quantised into five values and the tile reads as one flat sanded panel. Kept
                // modest — pushed harder the flakes blotch into a spongy travertine.
                col *= 0.92 + 0.16 * Noise.unit(Noise.mix(chip.cellId ^ 0x2))
                // A whisper of binder hairline — just enough to part two chips that drew the
                // same tone. This is ALL the boundary the material gets; the old bake made
                // this the entire structure.
                let hairline = 1 - smoothstep(0.0, 0.05, chip.f2 - chip.f1)
                col *= 1 - 0.10 * hairline

                // ── 3. Fibre striation along the chip's long axis (gu). Kept LOW: it competes
                // with the chip mosaic, and at full strength the tile reads as brushed
                // plywood rather than pressed granules.
                let fibre = Noise.fbmTiled(gu, gv * 9, baseCells: 22, octaves: 2, seed: sh ^ 0x33) - 0.5
                col *= 1 + 0.07 * fibre

                // ── 4. LENTICELS. Porosity is drawn PER CHIP *and* modulated by a broad
                // drift, so the pores arrive in the uneven bands the reference shows — some
                // patches almost solid pepper, others nearly clean.
                let pore = Noise.voronoiTiled(gu, gv, cells: poreCells, jitter: 1.0, seed: sh ^ 0x77)
                let drift = 0.55 + 1.20 * Noise.fbmTiled(gu * 3, gv * 3, baseCells: 5, octaves: 2, seed: sh ^ 0x99)
                let porosity = clamp01((0.19 + 0.56 * Noise.unit(Noise.mix(chip.cellId ^ 0x5EED))) * drift)
                var poreMask = 0.0
                if Noise.unit(Noise.mix(pore.cellId)) < porosity {
                    // Toroidal offset from the pore's own feature point, then squashed
                    // across the chip axis → a dash, not a dot.
                    var dx = gu - pore.center.x; dx -= dx.rounded()
                    var dy = gv - pore.center.y; dy -= dy.rounded()
                    let stretch = 1.8 + 1.7 * Noise.unit(pore.cellId ^ 0xA5)
                    let r = ((dx / stretch) * (dx / stretch) + dy * dy).squareRoot() * Double(poreCells)
                    let rad = 0.20 + 0.20 * Noise.unit(Noise.mix(pore.cellId ^ 0x3C))
                    poreMask = (1 - smoothstep(rad * 0.5, rad, r))
                        * (0.55 + 0.45 * Noise.unit(pore.cellId ^ 0x1D))   // not all pores are equally deep
                }
                col = mix(col, poreColor, poreMask * 0.90)

                // ── 5. Sparse burl flecks.
                let burl = Noise.voronoiTiled(gu, gv, cells: 24, jitter: 1.0, seed: sh ^ 0xB0)
                if Noise.unit(Noise.mix(burl.cellId ^ 0xBEEF)) < 0.10 {
                    let wob = 0.22 + 0.20 * (Noise.fbmTiled(gu * 6, gv * 6, baseCells: 6, octaves: 2, seed: sh ^ 0xB1) - 0.5)
                    col = mix(col, burlColor, (1 - smoothstep(wob * 0.45, wob, burl.f1)) * 0.75)
                }

                // ── 6. Per-tile tone + broad macro blotch. The JUMP across a joint is the
                // cue; the joint line itself is almost incidental.
                let macro = 0.94 + 0.12 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh ^ 0x22)
                let tileWarm = 0.97 + 0.06 * Noise.unit(Noise.mix(th ^ 0x1234))
                col *= (0.92 + 0.17 * Noise.unit(Noise.mix(th ^ 0x77))) * macro
                col.x *= tileWarm; col.z *= 2 - tileWarm

                // ── 7. Tile joint — tight, as cork tiles butt.
                let joint = max(1 - smoothstep(0.006, 0.016, min(lu, 1 - lu)),
                                1 - smoothstep(0.006, 0.016, min(lv, 1 - lv)))
                col *= 1 - 0.22 * joint

                ch.albedo[ch.idx(x, y)] = clampBand(col)

                // ── 8. Satin factory seal: EVEN, rising only where the seal thins.
                ch.roughness[ch.idx(x, y)] = clamp01(0.33 + poreMask * 0.30 + joint * 0.12
                    + (Noise.fbmTiled(u * 3, v * 3, baseCells: 3, octaves: 2, seed: sh ^ 0x44) - 0.5) * 0.07)

                // Relief: pores are real pits, chips sit at slightly different heights,
                // the joint is a groove.
                ch.height[ch.idx(x, y)] = clamp01(0.62 - poreMask * 0.40 - joint * 0.22
                    - hairline * 0.06 + 0.035 * fibre
                    + 0.05 * (Noise.unit(Noise.mix(chip.cellId ^ 0xF0)) - 0.5))
            }
        }
        ch.grainTangent = [Vec2](repeating: Vec2(0, 1), count: size * size)
        ch.clearcoat = 0.22
        ch.deriveNormals(strength: 4)
        // Sub-texel granule tooth under the seal — everything below ~4 mm lives here.
        addMicroDetail(&ch, seed: seed ^ 0xBB, baseCells: 84, strength: 0.55)
        return ch
    }

    /// Linen/cotton — plain-weave textile for curtains and upholstery. A soft matte woven
    /// cloth: at furniture-viewing distance you should read "fabric" without seeing a distinct
    /// mesh/screen grid. So the weave is a WHISPER — the cloth character is carried mostly by
    /// fine ROUGHNESS variation and the soft directional **sheen**, with only a faint albedo /
    /// normal ripple hinting at the interlaced warp/weft. The albedo weave contrast and the
    /// normal relief are deliberately tiny so the material stays a light, EVEN oatmeal at ANY
    /// surface orientation (a vertical back cushion must read the same light tone as a
    /// sun-facing seat — not a dark gridded screen), and the weave never beats into moiré on
    /// grazing panels. Every band is a smooth band-limited sinusoid (TAA-stable).
    public static func linen(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 107,
                             base: Vec3 = Vec3(0.76, 0.72, 0.65),
                             weaveContrast: Double = 1.0) -> MaterialChannels {
        // `weaveContrast` scales how far the interlace shows (albedo crown-vs-valley, roughness, normal relief). 1.0 is the
        // upholstery whisper — every existing caller, byte-identical. Drapery hangs in raking window light at arm's length,
        // where a bare cloth reads as a flat sheet unless the thread structure carries (see `draperyLinen`).
        var ch = MaterialChannels(size: size, category: .fabric)
        let sh = seed
        let tau = 2.0 * Double.pi
        // Threads across one tile. 20 over the 0.30 m upholstery tile → ~1.5 cm pitch: a coarse
        // hopsack read. Kept deliberately COARSE and LOW-CONTRAST so the periodic weave stays
        // far below the render's sampling Nyquist on grazing panels (no swirling moiré).
        let tp = 20.0
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let pu = u * tp, pv = v * tp
                // Rounded thread cross-sections (crown at cell centre, valley at the gap) —
                // smooth cosine ridges, integer thread count so they tile at the UV seam.
                let warpRidge = 0.5 - 0.5 * cos(tau * pu)   // vertical warp threads
                let weftRidge = 0.5 - 0.5 * cos(tau * pv)   // horizontal weft threads
                // Interlace: a smooth half-frequency checker picks which thread floats on top,
                // alternating every thread — the defining structure of a plain weave.
                let wOver = 0.5 + 0.5 * sin(Double.pi * pu) * sin(Double.pi * pv)  // 1 warp-over … 0 weft-over
                let crown = wOver * warpRidge + (1 - wOver) * weftRidge            // the raised on-top thread
                // Irregular slub (linen's signature uneven thread thickness) + fine fuzz so the
                // weave is perturbed, not a pristine grid (keeps it a stochastic texture).
                let slub = 0.965 + 0.05 * Noise.fbmTiled(u * tp * 0.5, v * tp * 0.5, baseCells: 8, octaves: 2, seed: sh ^ 0x71)
                let fuzz = Noise.fbmTiled(u * tp, v * tp, baseCells: 8, octaves: 2, seed: sh ^ 0x9C) - 0.5
                let macro = 0.955 + 0.06 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh ^ 0x7B)
                // WHISPER albedo weave: crown vs valley differ by only ~5% so the surface stays
                // an even light oatmeal — the weave is a texture hint, never a two-tone grid.
                let shade = 1.0 - 0.045 * weaveContrast + 0.05 * weaveContrast * crown
                // DH-1316: real linen's yarn — heathered per yarn, crossed by thick / thin slub streaks.
                let yarn = refinedFabricLook ? linenSlub(x: x, y: y, size: size, seed: sh ^ 0x51B)
                                             : (tone: 1.0, lift: 0.0, gloss: 0.0)
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * slub * shade * yarn.tone)
                // Roughness is high and EVEN: linen is a very matte cloth, so keep the mean up
                // (~0.90). The variation is LOW-FREQUENCY (broad smooth patches, NOT fine per-texel
                // speckle and NOT the periodic weave) so it gives subtle fabric life without a grain
                // that reads as a dirty mottle on a dimly-lit panel.
                ch.roughness[ch.idx(x, y)] = clamp01(0.90 - crown * 0.02 * weaveContrast - yarn.gloss
                    + (Noise.fbmTiled(u * 2.5, v * 2.5, baseCells: 2, octaves: 2, seed: sh ^ 0x3F) - 0.5) * 0.18)
                ch.height[ch.idx(x, y)] = clamp01(0.40 + 0.34 * crown * slub + 0.05 * fuzz + yarn.lift)
            }
        }
        // A MODERATE cloth sheen — the soft directional nap glow real linen has. Kept well below
        // velvet grade so it reads as matte upholstery, not satin: the fabric character is carried
        // mostly by the albedo weave + roughness variation, with sheen as the finishing soft glow.
        ch.sheen = 0.30
        ch.diffuseRoughness = 0.50                  // DH-1301 — a flat plain weave: matte, the yarns' crowns smooth
        // Nap width kept at the library DEFAULT (0.30) on purpose: a flat plain weave has a
        // medium, unremarkable sheen breadth, and this is the fabric the cloth-sheen render gate
        // measures — leaving it default keeps that frame byte-identical (DH-0081).
        ch.sheenRoughness = 0.30
        ch.grainTangent = [Vec2](repeating: Vec2(1, 0), count: size * size)
        ch.deriveNormals(strength: 0.1 * weaveContrast)            // near-flat — the weave is only a hint, so a
                                                   // grazing back cushion stays even soft cloth,
                                                   // never a specular mesh/screen grid
        // Fine sub-thread fuzz — the linen tooth at grazing range, kept to a whisper (a stronger
        // detail normal sparkles into a dirty grain on a dimly-lit grazing panel).
        //
        // S2.4: that 0.03 is a limit on the NORMAL, not a statement that linen's weave is
        // 3 % deep — and the occlusion has no sparkle failure mode, so it takes the weave's
        // real depth. Measured at the grazing gate: 1.01 (normal-only) → 1.04 inheriting 0.03,
        // → 1.2× at the weave's own strength, which is what a linen cushion should do.
        // DH-1339: a real plain weave in the detail band — 8 texels (~0.55 mm) per thread, warp over weft alternating, each
        // thread its own width, with a slow slub along it — plus a thin haze of loose fibres over the top. The normal stays
        // modest (the old note: a strong detail normal sparkles on a dim grazing panel); the occlusion carries the depth.
        addFibreDetail(&ch, seed: seed ^ 0xBC, fibres: [
            .init(count: 2600, length: 34, width: 2.0, angle: 0, spread: .pi, layered: false, widthJitter: 0.5)],
            baseCells: 100, weave: .init(pitch: 8, depth: 0.8, threadJitter: 0.35, slub: 0.6),
            strength: 0.25 * min(weaveContrast, 2), occlusionStrength: min(0.55 * weaveContrast, 0.95))
        return ch
    }

    /// Linen for curtains: the same plain weave with its interlace pushed (`weaveContrast` 3) so the thread structure reads
    /// on a hanging panel. A separate catalog entry — sofas, placemats and shades keep the upholstery whisper.
    public static func draperyLinen(size: Int = MaterialGenerator.bakeSize) -> MaterialChannels {
        linen(size: size, seed: 109, weaveContrast: 3.0)
    }

    /// Top-grain leather — smooth hide with a fine pore dimple pattern and subtle
    /// pull-up effect at creases (darker at recesses). Good for upholstery and headboards.
    public static func leather(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 127,
                               base: Vec3 = Vec3(0.56, 0.34, 0.20)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .fabric)
        let sh = seed
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let macro = 0.88 + 0.24 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh)
                // Fine pore dimples — very high frequency Voronoi cell borders
                let pore = Noise.voronoiTiled(u, v, cells: 38, jitter: 0.8, seed: sh ^ 0x22)
                let poreDark = smoothstep(0.0, 0.45, pore.f2 - pore.f1)     // DH-1339: was 0.12 — a crisp 8 mm polygon network (dried mud); now a soft mottle
                // Pull-up: where creases occur (low macro noise), leather lightens
                let pullUp = smoothstep(0.6, 0.9, macro) * 0.12
                let poreDepth = (1 - poreDark) * 0.02      // DH-1339: was 0.08 — the 8 mm cells read as dried mud once the real grain exists
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * (0.93 + poreDark * 0.07 + pullUp))
                ch.roughness[ch.idx(x, y)] = clamp01(0.45 + poreDark * 0.22
                    + (Noise.fbmTiled(u, v, baseCells: 8, octaves: 2, seed: sh ^ 0x55) - 0.5) * 0.08)
                ch.height[ch.idx(x, y)] = clamp01(0.55 - poreDepth
                    + Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: sh ^ 0x66) * 0.04)
            }
        }
        ch.clearcoat = 0.08
        ch.deriveNormals(strength: 4)
        // Fine pebbled top-grain — the dimpled hide texture reads at grazing angles; without
        // it, top-grain leather flattens to a plastic vinyl look.
        // DH-1339: real pebbled top-grain — ~1.5 mm irregular cells with sunken creases, fine secondary grain, sparse pores.
        addPebbleGrainDetail(&ch, seed: seed ^ 0xBF, cells: 26, strength: 0.9, occlusionStrength: 0.6)
        return ch
    }

    // MARK: – Wallpaper (customizable, colorway variants)

    /// The wall-covering pattern a `WallpaperParams` selects. Kept small and tasteful —
    /// each is fully procedural and tileable.
    public enum WallpaperPattern: String, Equatable, Hashable, Sendable, Codable, CaseIterable {
        /// Woven natural-fibre cloth: fine warp/weft threads, organic slub. Reads as the
        /// matte, textured "grasscloth" wall covering. The TextureAudit-safe default —
        /// its fibre structure gives natural roughness variation for free.
        case grasscloth
        /// Two-tone vertical stripe (base ground + accent band), with a soft paper-grain
        /// roughness so it never trips the flat tell.
        case stripe
        /// Soft tonal damask motif — a low-frequency ogee-ish pattern in accent over the
        /// base ground, subtle and self-coloured (tone-on-tone).
        case damask
    }

    /// Customization spine for the wallpaper material — the §7 pattern Roominate exposed as
    /// a "variant", here generalized: a base ground colour, an accent, a pattern selector,
    /// and a repeat scale. Defaults reproduce a warm neutral grasscloth.
    public struct WallpaperParams: Equatable, Hashable, Sendable, Codable {
        /// The dominant ground colour (paper/fibre base). Linear, clamped to the band.
        public var baseColor: Vec3
        /// The secondary colour: stripe band, damask motif, or fibre-highlight tint.
        public var accentColor: Vec3
        /// Which procedural pattern to bake.
        public var pattern: WallpaperPattern
        /// Repeat-scale multiplier: 1 = the design repeat below; 0.5 = finer, 2 = coarser.
        /// Quantized to an integer thread/band count internally so the tile stays seamless.
        public var patternScale: Double
        /// Redraws the fibre/grain scatter.
        public var seed: UInt64

        public init(baseColor: Vec3 = Vec3(0.74, 0.70, 0.61),
                    accentColor: Vec3 = Vec3(0.60, 0.55, 0.46),
                    pattern: WallpaperPattern = .grasscloth,
                    patternScale: Double = 1, seed: UInt64 = 7) {
            self.baseColor = clampBand(baseColor)
            self.accentColor = clampBand(accentColor)
            self.pattern = pattern
            self.patternScale = patternScale
            self.seed = seed
        }

        /// Tolerant decode — any missing key falls back to the default look, so a saved
        /// document from before a field existed still loads (pre-release, no migration shims).
        public init(from decoder: Decoder) throws {
            let d = try decoder.container(keyedBy: CodingKeys.self)
            let def = WallpaperParams()
            baseColor   = clampBand(try d.decodeIfPresent(Vec3.self, forKey: .baseColor) ?? def.baseColor)
            accentColor = clampBand(try d.decodeIfPresent(Vec3.self, forKey: .accentColor) ?? def.accentColor)
            pattern     = try d.decodeIfPresent(WallpaperPattern.self, forKey: .pattern) ?? def.pattern
            patternScale = try d.decodeIfPresent(Double.self, forKey: .patternScale) ?? def.patternScale
            seed        = try d.decodeIfPresent(UInt64.self, forKey: .seed) ?? def.seed
        }

        /// Curated colorways (the variant presets the library modal reads).
        public static let presets: [(name: String, params: WallpaperParams)] = [
            ("Natural Grasscloth", .init()),
            ("Flax Linen",  .init(baseColor: Vec3(0.80, 0.77, 0.69), accentColor: Vec3(0.66, 0.62, 0.53),
                                  pattern: .grasscloth)),
            ("Sage Stripe", .init(baseColor: Vec3(0.80, 0.81, 0.74), accentColor: Vec3(0.52, 0.58, 0.48),
                                  pattern: .stripe)),
            ("Slate Damask", .init(baseColor: Vec3(0.40, 0.45, 0.52), accentColor: Vec3(0.52, 0.57, 0.63),
                                   pattern: .damask)),
            ("Blush Damask", .init(baseColor: Vec3(0.82, 0.72, 0.70), accentColor: Vec3(0.72, 0.60, 0.58),
                                   pattern: .damask)),
        ]
    }

    /// Wallpaper: a tasteful, tileable wall covering with three pattern modes (grasscloth /
    /// stripe / damask) and a two-colour customization spine. Roughness is varied spatially
    /// in every mode (fibre/grain/paper micro-noise) so it clears the flat-roughness tell #3,
    /// and all patterns use integer-multiple `fbmTiled` frequencies so the wrap is seamless.
    public static func wallpaper(size: Int = MaterialGenerator.bakeSize, params: WallpaperParams = .init()) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .wallpaper)
        let sh = params.seed
        // Quantize the repeat to integers so the tile stays seamless at any scale.
        let scale = max(0.35, min(3.0, params.patternScale))
        let stripeBands = max(2, Int((6.0 / scale).rounded()))          // vertical stripe count
        let threadFreq  = max(8,  Int((28.0 / scale).rounded()))        // grasscloth thread count
        let damaskFreq  = max(2,  Int((4.0 / scale).rounded()))         // damask motif repeats
        // Even the stripe/damask grounds get a faint paper grain so roughness varies.
        let base = params.baseColor, accent = params.accentColor

        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)

                // Paper grain shared by all modes — keeps roughness spatially varied.
                let grain = Noise.fbmTiled(u * 18, v * 18, baseCells: 4, octaves: 2, seed: sh ^ 0x4D)
                let macro = 0.92 + 0.16 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh ^ 0x7B)

                var albedo: Vec3
                var rough: Double
                var height: Double

                switch params.pattern {
                case .grasscloth:
                    // Woven natural fibre: horizontal weft threads dominate (the grasscloth
                    // look), with a finer warp cross-thread. Integer thread frequencies tile.
                    let weft = Noise.fbmTiled(u * 3, v * Double(threadFreq), baseCells: 4, octaves: 2, seed: sh)
                    let warp = Noise.fbmTiled(u * Double(threadFreq), v * 3, baseCells: 4, octaves: 2, seed: sh ^ 0x5E)
                    // Which fibre is raised at each crossing (tileable, no discrete floor()).
                    let cross = Noise.fbmTiled(u * Double(threadFreq), v * Double(threadFreq),
                                               baseCells: 4, octaves: 1, seed: sh ^ 0x3A)
                    let weave = cross > 0.5 ? weft : warp
                    let slub  = 0.95 + 0.10 * Noise.fbmTiled(u * 5, v * 5, baseCells: 2, octaves: 2, seed: sh ^ 0x71)
                    // Fibre colour drifts between base and accent along the weave.
                    let fibre = mix(base, accent, 0.30 + 0.40 * weave)
                    albedo = clampBand(fibre * slub * macro * (0.82 + 0.30 * weave))
                    rough  = clamp01(0.78 - weave * 0.16 + (grain - 0.5) * 0.10)
                    height = clamp01(0.40 + weave * 0.40)

                case .stripe:
                    // Smooth vertical two-tone stripe: a cosine band so the boundary is soft
                    // (no hard floor() wrap seam). `stripeBands` complete periods across u.
                    let band = 0.5 + 0.5 * cos(2.0 * Double.pi * Double(stripeBands) * u)
                    let stripeMask = smoothstep(0.35, 0.65, band)
                    let col = mix(base, accent, stripeMask)
                    // Faint vertical paper-pulp striation in the accent band.
                    let pulp = Noise.fbmTiled(u * Double(stripeBands * 2), v * 4, baseCells: 4, octaves: 2, seed: sh ^ 0x22)
                    albedo = clampBand(col * macro * (0.96 + 0.06 * pulp))
                    rough  = clamp01(0.62 + stripeMask * 0.06 + (grain - 0.5) * 0.12)
                    height = clamp01(0.50 + (grain - 0.5) * 0.10 + stripeMask * 0.04)

                case .damask:
                    // Tone-on-tone ogee-ish motif from two phase-shifted low-frequency
                    // sinusoids (a classic damask lattice), softened to a tonal wash.
                    let f = Double(damaskFreq)
                    let a = sin(2 * .pi * f * u) * sin(2 * .pi * f * v)
                    let b = cos(2 * .pi * f * (u + 0.5)) * cos(2 * .pi * f * (v + 0.25))
                    let motif = smoothstep(0.15, 0.75, 0.5 + 0.5 * (0.6 * a + 0.4 * b))
                    let col = mix(base, accent, motif * 0.85)
                    albedo = clampBand(col * macro * (0.97 + 0.05 * grain))
                    // Damask printed ink is slightly more matte than the paper ground.
                    rough  = clamp01(0.60 + motif * 0.10 + (grain - 0.5) * 0.12)
                    height = clamp01(0.50 + motif * 0.06 + (grain - 0.5) * 0.08)
                }

                ch.albedo[ch.idx(x, y)] = albedo
                ch.roughness[ch.idx(x, y)] = rough
                ch.height[ch.idx(x, y)] = height
            }
        }
        // Wallpaper is matte paper — no clearcoat. Grasscloth fibre gives the most relief.
        ch.deriveNormals(strength: params.pattern == .grasscloth ? 3 : 2)
        // Paper tooth — the fine matte fibre grain of the wall covering at grazing range.
        addMicroDetail(&ch, seed: sh ^ 0xC1, baseCells: 100, strength: 0.40)
        return ch
    }
}
