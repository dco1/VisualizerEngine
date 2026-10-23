import simd
import Foundation

extension MaterialGenerator {
    // Oak (the §5a directional wood-grain example) lives in `MaterialGenerator+Wood.swift`
    // alongside the shared `woodGrain` engine — the wood family's directional logic is the
    // most intricate in the library, so it gets its own focused file.

    // TILE — the whole family (square / rectangle / plank, any bond, any size, any colour,
    // grout colour, edge profile, finish) lives in `MaterialGenerator+Tile.swift`, alongside the
    // `UnitPatternParams` spine it shares with `unitPavers`. It replaced the near-duplicate pair
    // that used to sit here: `ceramicTile` (square, stack-bond) and `wallTile` (running bond, and
    // 3:1 or 2:1 depending on which entry point you called — customizing a wall tile silently
    // changed its proportion). One generator, one layout, one proportion.

    /// White **stoneware** — matte off-white ceramic tableware (a dinner plate, not a tiled
    /// wall). Distinct from `ceramicTile`: NO grout grid, a single continuous glazed body with
    /// gentle firing tone variation and a fine speckle so it isn't a flat white. Satin, not
    /// glossy (real stoneware is a low-sheen glaze), so `clearcoat` stays modest and roughness
    /// mid-range. Roughness varies spatially (speckle + macro) so it clears TextureAudit's
    /// flat-roughness tell; albedo carries enough macro contrast to clear the flat-colour tell.
    public static func stoneware(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 151,
                                 body: Vec3 = Vec3(0.90, 0.89, 0.86)) -> MaterialChannels {
        // `.ceramic`, not `.tile`: this is a fired clay BODY with no grid on it. Filing it with
        // unit masonry is what put a grout-less tableware glaze in the floor and wall pickers.
        var ch = MaterialChannels(size: size, category: .ceramic)
        let sh = seed
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                // Broad firing tone: slow warm/cool drift across the body (macro contrast source).
                let macro = 0.955 + 0.09 * (Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh) - 0.5)
                // Fine mineral speckle — sparse darker flecks from the clay body (reads as stoneware,
                // not bone china). Sharp Voronoi cell centres kept dark and rare.
                let spk = Noise.voronoiTiled(u, v, cells: 40, jitter: 0.9, seed: sh ^ 0x2C)
                let fleck = pow(max(0, 1 - (spk.f2 - spk.f1) * 5), 8) * 0.10   // rare dark specks
                let tone = macro * (1 - fleck)
                ch.albedo[ch.idx(x, y)] = clampBand(body * tone)
                // Satin glaze roughness ~0.34, varied by speckle (flecks slightly rougher) and a
                // fine orange-peel band so the surface isn't a mirror-flat plane.
                let peel = (Noise.fbmTiled(u, v, baseCells: 24, octaves: 3, seed: sh ^ 0x51) - 0.5) * 0.06
                ch.roughness[ch.idx(x, y)] = clamp01(0.34 + fleck * 1.6 + peel)
                ch.height[ch.idx(x, y)] = clamp01(0.55 - fleck * 2
                    + (Noise.fbmTiled(u, v, baseCells: 6, octaves: 2, seed: sh ^ 0x71) - 0.5) * 0.06)
            }
        }
        ch.clearcoat = 0.12                        // satin glaze — soft sheen, not mirror
        ch.deriveNormals(strength: 3)
        addMicroDetail(&ch, seed: sh ^ 0xC7, baseCells: 90, strength: 0.35,   // fine glaze micro-relief
                       occlusionStrength: 0.35 * glazeDetailOcclusionShare)
        return ch
    }

    // ── Glazed sanitaryware (toilet / bathtub / vanity basin) TASTE DIALS ─────────
    // Flagged for Danny: the tone + gloss of the ceramic. Conservative clean glossy
    // white. These are the ONLY knobs to reach for — the generator derives everything
    // from them, so there is one place to warm/cool the tone or dial the wet sheen.
    //   • `ceramicAlbedo`    — the fired-glaze body colour. A hair cool of neutral so it
    //                          reads as bright bathroom white, not cream. Warm it toward
    //                          (0.91,0.90,0.88) for an "almond/biscuit" sanitaryware.
    //   • `ceramicClearcoat` — the glaze wet-spec lobe [0,1]. Higher = wetter, glossier
    //                          highlight (real fired glaze is very glossy — glossier than
    //                          the satin `stoneware` tableware, 0.12). 0.60 is a clean,
    //                          not-mirror sheen.
    //   • `ceramicRoughness` — base glaze roughness. Low = glossy. Lower toward ~0.06 for
    //                          a wetter mirror; raise toward ~0.20 for a soft matte glaze.
    // DH-1397: was (0.905, 0.910, 0.915) — brighter than any real glaze (white porcelain/sanitaryware
    // reads 0.80-0.88 linear, nothing diffuse >0.9).
    public static let ceramicAlbedo    = Vec3(0.855, 0.860, 0.865)
    public static let ceramicClearcoat = 0.60
    public static let ceramicRoughness = 0.11

    /// Glazed **sanitaryware** — the wet, glossy white ceramic of a toilet / bathtub /
    /// lavatory basin. Distinct from BOTH siblings: `ceramicTile` is a tiled floor/wall
    /// surface WITH a grout grid, and `stoneware` is a SATIN matte tableware glaze. This is
    /// a seamless, continuous, high-gloss glazed body — no grout, a much stronger clearcoat
    /// and a lower roughness than stoneware — with a VERY subtle large-scale glaze value
    /// mottle + a fine orange-peel micro-relief so a large tub or bowl doesn't read as a
    /// dead-flat plastic sheet (the flat-white → plastic tell). Roughness varies spatially
    /// (orange-peel band) so it clears TextureAudit's flat-roughness tell; the glaze mottle
    /// carries enough macro contrast to clear the flat-colour tell. Tone/gloss come from the
    /// three TASTE DIALS above.
    public static func ceramic(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 197,
                               body: Vec3 = MaterialGenerator.ceramicAlbedo) -> MaterialChannels {
        // `.ceramic`, not `.tile` — see `stoneware`. Both families are COHERENT for anti-tiling,
        // so the pixels and the hex-blend decision are unchanged; only the filing is honest now.
        var ch = MaterialChannels(size: size, category: .ceramic)
        let sh = seed
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                // Two-scale glaze value mottle: a broad firing drift + a medium unevenness.
                // Kept faint (the surface is a clean bright white) but enough tonal spread to
                // clear the flat-colour tell — a real fired glaze is never perfectly uniform.
                let broad  = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: sh) - 0.5
                let medium = Noise.fbmTiled(u, v, baseCells: 7, octaves: 2, seed: sh ^ 0x22) - 0.5
                let mottle = 1.0 + 0.20 * broad + 0.12 * medium
                ch.albedo[ch.idx(x, y)] = clampBand(body * mottle)
                // Glossy glaze roughness, varied by a coarse glaze-thickness drift + a fine
                // orange-peel band so the surface isn't a mirror-flat plane (clears the
                // flat-roughness tell) while still reading as a wet glaze. Glaze pools glossier
                // where the value mottle is darker/thicker (roughness dips with the broad term).
                let peel = (Noise.fbmTiled(u, v, baseCells: 22, octaves: 3, seed: sh ^ 0x53) - 0.5) * 0.10
                ch.roughness[ch.idx(x, y)] = clamp01(ceramicRoughness + peel - 0.10 * broad)
                ch.height[ch.idx(x, y)] = clamp01(0.5
                    + (Noise.fbmTiled(u, v, baseCells: 8, octaves: 2, seed: sh ^ 0x77) - 0.5) * 0.05)
            }
        }
        ch.clearcoat = ceramicClearcoat            // strong glaze clearcoat — the wet glossy sheen
        ch.deriveNormals(strength: 2)
        addMicroDetail(&ch, seed: sh ^ 0xD9, baseCells: 100, strength: 0.30,   // fine glaze orange-peel
                       occlusionStrength: 0.30 * glazeDetailOcclusionShare)
        return ch
    }


    /// Marble — **domain-warped veining** (§5): two fBm fields warp the sample point,
    /// then a ridged function of the warped field draws the thin mineral veins (Inigo
    /// Quilez's `f(p + h(p))` recipe). A near-white polished stone with darker veins; the
    /// warp keeps the veins organic and non-repeating across the tile.
    /// **How far marble's veins are stretched along their bedding direction.** Integer, because
    /// `Noise.valueTiled` wraps on its cell count and a fractional stretch would break the tile
    /// (and the seam audit with it). 3 gives clearly-directional trains that still wander.
    public static let marbleBeddingStretch: Double = 3
    /// Ridge value at which a vein's dark CORE begins. Higher = thinner, sharper veins. The whole
    /// difference between "veins" and "smoke" lives in this being a threshold at all.
    public static let marbleVeinCoreEdge: Double = 0.94

    public static func marble(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 5,
                              base: Vec3 = Vec3(0.86, 0.85, 0.83),
                              veinColor: Vec3 = Vec3(0.34, 0.33, 0.36)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .stone)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)

                // **Bedding-aligned domain.** Marble is metamorphosed limestone: its veins are
                // deformed SEDIMENTARY BEDS, so they share a dominant direction and run in
                // roughly parallel trains. Sampling an isotropic field gave equal energy in
                // every direction, which is what produced closed swirls and cell-like curls
                // rather than veins. Compressing v by an integer factor elongates every feature
                // along u; the factor is integer so `valueTiled`'s wrap — and the seam audit —
                // still hold.
                let bv = v * marbleBeddingStretch

                // **Warp is 0.18, not 0.6.** A domain warp of 0.6 displaces a sample by 60 % of
                // the tile, which is far larger than the vein spacing it is meant to perturb: it
                // does not bend the veins, it shuffles the field into an unstructured cloud, and
                // that cloud IS the airbrushed-fog read. This amount wanders a vein by a few per
                // cent of the tile — the geological "wobble" — and leaves the train intact.
                let qx = Noise.fbmTiled(u, v, baseCells: 2, octaves: 4, seed: seed)
                let qy = Noise.fbmTiled(u, v, baseCells: 2, octaves: 4, seed: seed ^ 0x71)
                let wu = u + 0.18 * (qx - 0.5)
                let wv = bv + 0.35 * (qy - 0.5)

                // **A vein is an EDGE, so it needs a threshold, not a power curve.** `pow(ridge, 6)`
                // has no crisp boundary anywhere — it falls off smoothly over the whole cell, and
                // summing four such layers is precisely how the panel turned into grey smoke. A
                // real vein has a dark, narrow, sharply-bounded core with a diffuse mineral BLEED
                // either side, so it is authored as those two terms.
                let ridge = 1 - abs(Noise.fbmTiled(wu, wv, baseCells: 5, octaves: 5,
                                                   seed: seed ^ 0x9C) * 2 - 1)
                let core  = smoothstep(marbleVeinCoreEdge, 0.999, ridge)
                // The bleed is a NARROW mineral halo hugging the core, not a tonal mass. At
                // smoothstep(0.52, 0.95) it covered roughly 40 % of the slab in soft grey and the
                // swatch read as camouflage: a marble slab is overwhelmingly pale matrix, and the
                // veins are the exception in it.
                let bleed = smoothstep(0.82, 0.985, ridge) * 0.22

                // **Capillaries hang OFF the primary veins.** Gating the fine network by the
                // primary field's own proximity-to-crest is what makes the result read as one
                // branching system instead of two independent noise layers laid over each other
                // (the old vein2/vein3 were ungated, so they crossed the mains at random and
                // added yet another isotropic haze).
                let ridge2 = 1 - abs(Noise.fbmTiled(wu * 3, wv * 3, baseCells: 3, octaves: 4,
                                                    seed: seed ^ 0x33) * 2 - 1)
                let cap = smoothstep(0.93, 1.0, ridge2) * smoothstep(0.45, 0.88, ridge) * 0.65

                // Slow clouding — the pale tonal drift of the matrix. Kept faint (±2.5 %): the
                // old ±5 % macro plus a 0.25-amplitude broad "secondary vein" was most of the wash.
                let macro = 0.975 + 0.05 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3,
                                                          seed: seed ^ 0x2D)
                let mask = clamp01(core + bleed + cap)
                ch.albedo[ch.idx(x, y)] = clampBand(mix(base, veinColor, mask) * macro)

                // Polished stone: low roughness overall, and the veins a touch rougher — softer
                // calcite polishes to a slightly duller face than the matrix, which is what makes
                // a real slab's veins catch the light differently rather than only being darker.
                ch.roughness[ch.idx(x, y)] = clamp01(0.12 + mask * 0.14
                    + (Noise.fbmTiled(u, v, baseCells: 28, octaves: 2, seed: seed ^ 0x4E) - 0.5) * 0.03)
                // Differential polish: the vein sits very slightly proud-of/below the matrix.
                ch.height[ch.idx(x, y)] = clamp01(0.5 - mask * 0.22)
            }
        }
        ch.clearcoat = 0.30                        // polished marble
        ch.deriveNormals(strength: 3)
        addMicroDetail(&ch, seed: seed ^ 0xF4, baseCells: 72, strength: 0.5)   // honed micro-relief
        return ch
    }

    /// Polished concrete — low-reflectance grey whose defining trait is **spatially
    /// varying roughness** (§5): broad fBm blotches of sheen over a matte field, plus
    /// fine pores and the odd darker stain. A light clearcoat for the polished sheen.
    public static func concrete(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 7,
                                base: Vec3 = Vec3(0.56, 0.56, 0.55)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .concrete)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let blotch = Noise.fbmTiled(u, v, baseCells: 3, octaves: 5, seed: seed)
                let stain = pow(Noise.fbmTiled(u, v, baseCells: 5, octaves: 4, seed: seed ^ 0x1A), 3) * 0.5
                let pore = (Noise.fbmTiled(u, v, baseCells: 40, octaves: 2, seed: seed ^ 0x2B) - 0.5)

                // Board-form lines — faint horizontal seams where the form boards met (integer
                // count tiles in v), and sparse cast bug-holes (air pockets). Both are the
                // "poured concrete" tells Roominate had; tiling Worley + integer cos stay seamless.
                let form = pow(0.5 - 0.5 * cos(2 * .pi * v * 6), 10)    // narrow dark line per board seam
                let bug = Noise.voronoiTiled(u, v, cells: 9, jitter: 0.9, seed: seed ^ 0x77)  // 0.9 (not 1.0): keeps the ±1-cell search torus-consistent at the tile wrap (no seam)
                let hole = 1 - smoothstep(0.0, 0.028, bug.f1)          // 1 inside a tiny pit

                let tone = (0.90 + 0.14 * blotch) * (1 - 0.18 * stain) * (1 - 0.10 * form) * (1 - 0.45 * hole)
                ch.albedo[ch.idx(x, y)] = clampBand(base * tone)
                // the signature: roughness drifts spatially (polished patches vs matte) + rough seams/holes
                ch.roughness[ch.idx(x, y)] = clamp01(0.25 + 0.22 * blotch + 0.18 * stain + pore * 0.05
                    + 0.15 * form + 0.30 * hole)
                ch.height[ch.idx(x, y)] = clamp01(0.5 + pore * 0.4 - stain * 0.1 - 0.15 * form - 0.5 * hole)
            }
        }
        ch.clearcoat = 0.20                        // polished-concrete sheen
        ch.deriveNormals(strength: 3)
        addMicroDetail(&ch, seed: seed ^ 0xF5, baseCells: 88, strength: 0.7)   // sand/grit micro-tooth
        return ch
    }

    /// **Terracotta (unglazed fired clay)** — the warm earthenware a classic garden planter is
    /// thrown from (DH-0124). Categorised `.stone` (a fired-earth dielectric that belongs with
    /// the natural-mineral surfaces and de-repeats stochastically), never glazed: matte, no
    /// clearcoat, an orange-red body mottled by uneven firing with the odd darker scorch and a
    /// scatter of tiny surface pinholes (the air pockets a low-fire clay always shows). The
    /// mottle carries the macro albedo variation the dielectric tone tell requires.
    public static func terracotta(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 223,
                                  base: Vec3 = Vec3(0.60, 0.31, 0.20)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .stone)
        let scorch = Vec3(0.44, 0.22, 0.14)          // darker over-fired blush
        let light  = Vec3(0.70, 0.42, 0.29)          // lighter under-fired flush
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let fire = Noise.fbmTiled(u, v, baseCells: 3, octaves: 5, seed: seed)          // firing mottle
                let blush = pow(Noise.fbmTiled(u, v, baseCells: 4, octaves: 4, seed: seed ^ 0x31), 2.5) * 0.7
                let pit = Noise.voronoiTiled(u, v, cells: 22, jitter: 0.9, seed: seed ^ 0x6D)
                let hole = 1 - smoothstep(0.0, 0.030, pit.f1)                                   // 1 inside a pinhole
                var c = mix(base * (0.88 + 0.20 * fire), light, blush)
                c = mix(c, scorch, pow(1 - fire, 3) * 0.5)
                c = c * (1 - 0.35 * hole)                                                       // pinholes read darker
                ch.albedo[ch.idx(x, y)] = clampBand(c)
                ch.roughness[ch.idx(x, y)] = clamp01(0.70 + 0.10 * fire + 0.12 * hole
                    + (Noise.fbmTiled(u, v, baseCells: 26, octaves: 2, seed: seed ^ 0x4E) - 0.5) * 0.08)
                ch.height[ch.idx(x, y)] = clamp01(0.55 + (fire - 0.5) * 0.3 - 0.5 * hole)
            }
        }
        ch.clearcoat = 0.0                          // unglazed — no gloss
        ch.diffuseRoughness = 0.80                  // DH-1301 — unglazed low-fire clay: chalky, strongly matte
        ch.deriveNormals(strength: 2.5)
        addMicroDetail(&ch, seed: seed ^ 0xE7, baseCells: 88, octaves: 2, strength: 0.80)   // clay grain tooth
        return ch
    }

    /// Granite — the aggregate family (§5) at fine grain: dense angular feldspar/quartz/biotite
    /// grains (Voronoi cells, per-grain color), rare **mica** flecks that polish to a near-mirror
    /// sparkle, a polished clearcoat, and tight micro relief. A common counter + floor stone.
    public static func granite(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 11) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .stone)
        let palette: [Vec3] = [
            Vec3(0.78, 0.72, 0.66),   // feldspar tan
            Vec3(0.86, 0.84, 0.82),   // quartz white-grey
            Vec3(0.20, 0.19, 0.20),   // biotite black
            Vec3(0.62, 0.55, 0.52),   // pink-grey feldspar
            Vec3(0.50, 0.52, 0.55),   // grey
            Vec3(0.70, 0.60, 0.56),   // rose
        ]
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Half-cell phase offset (grainPhase) so the tile boundary (x=0) lands at a grain
                // CENTRE, not on the Voronoi cell lattice line. The texture is already perfectly
                // toroidal (voronoiTiled is periodic), but a lattice-aligned edge makes the wrap
                // column a grain-boundary column while the interior baseline is mostly smooth
                // within-grain columns — inflating TextureAudit's seam ratio with no visible seam.
                // Phasing the sample off the lattice makes the wrap column representative → seam≈1.
                let cell = Noise.voronoiTiled(u + Self.grainPhase, v + Self.grainPhase,
                                              cells: 26, jitter: 1.0, seed: seed)  // fine angular grains
                var grain = palette[Int(cell.cellId % UInt64(palette.count))]
                grain *= 0.82 * (0.90 + 0.18 * Noise.unit(Noise.mix(cell.cellId)))        // per-grain value; 0.82 = DH-1397 (polished granite 0.1-0.5, was 0.56)
                let macro = 0.95 + 0.10 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed ^ 0x2D)
                ch.albedo[ch.idx(x, y)] = clampBand(grain * macro)

                // Per-grain roughness: dark biotite reads rough, light quartz/feldspar polished —
                // real spatial variation (defeats the flat-roughness tell #3), not one scalar.
                let grainLuma = (grain.x + grain.y + grain.z) / 3
                let baseR = mix(0.10, 0.34, clamp01(1 - grainLuma))
                // Mica: rare bright flecks that polish to a near-mirror (very low roughness).
                let sparkle = Noise.fbmTiled(u, v, baseCells: 90, octaves: 2, seed: seed ^ 0x33)
                let mica = smoothstep(0.80, 0.92, sparkle)
                let pore = (Noise.fbmTiled(u, v, baseCells: 30, octaves: 2, seed: seed ^ 0x5C) - 0.5) * 0.03
                ch.roughness[ch.idx(x, y)] = clamp01(baseR - mica * 0.08 + pore)
                // Grain boundaries groove slightly; fine micro relief otherwise.
                let seam = 1 - smoothstep(0.0, 0.06, cell.f2 - cell.f1)
                ch.height[ch.idx(x, y)] = clamp01(0.55 - seam * 0.20
                    + Noise.fbmTiled(u, v, baseCells: 40, octaves: 2, seed: seed ^ 0x3F) * 0.03)
            }
        }
        ch.clearcoat = 0.40                        // polished granite — wet sheen
        ch.deriveNormals(strength: 4)
        addMicroDetail(&ch, seed: seed ^ 0xF6, baseCells: 96, strength: 0.5)   // crystal micro-sparkle relief
        return ch
    }

    /// Flagstone / slate paving — an irregular crazy-paving of large stone flags separated by
    /// dark recessed joints. Each flag carries its own grey-blue/charcoal slate tone (per-cell
    /// jitter), gentle within-flag cleft mottle, and an outdoor-matte finish with a faint damp
    /// sheen (low clearcoat). The Voronoi cell partition gives the irregular flag shapes; the
    /// near-boundary band (`f2 - f1` small) is the mortar joint — recessed and darker. Roughness
    /// varies (joint vs. flag face vs. cleft), macro tone comes from per-flag colour variation.
    public static func stone(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 137) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        // Slate family: cool desaturated grey, ranging charcoal → pale bluish grey per flag.
        let dark = Vec3(0.20, 0.21, 0.23)
        let light = Vec3(0.52, 0.53, 0.55)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Irregular flag partition. jitter 0.95 → natural crazy-paving cell shapes.
                let cell = Noise.voronoiTiled(u, v, cells: 6, jitter: 0.95, seed: seed)
                // Joint band: where f2 - f1 is small we're near a cell boundary (mortar gap).
                let edge = cell.f2 - cell.f1
                let joint = smoothstep(0.10, 0.02, edge)                 // 1 in joint … 0 on flag
                // Per-flag base tone (stable per-cell hash) → macro lightness variation.
                let flagTone = Double(cell.cellId >> 8 & 0xFFFF) / 65535.0
                // Within-flag cleft / riven slate mottle (mid + fine).
                let cleft = Noise.fbmTiled(u, v, baseCells: 10, octaves: 3, seed: seed ^ 0x3B)
                let fine  = Noise.fbmTiled(u, v, baseCells: 40, octaves: 2, seed: seed ^ 0xE2)

                // Flag colour: dark↔light by per-flag tone, plus cleft mottle.
                var flag = dark + (light - dark) * flagTone
                flag += Vec3(0.06, 0.06, 0.07) * (cleft - 0.5)
                flag += Vec3(0.03, 0.03, 0.03) * (fine - 0.5)
                // Joint: dark recessed mortar (cool grey, near the dielectric floor).
                let jointCol = Vec3(0.10, 0.105, 0.11)
                let col = flag + (jointCol - flag) * joint
                ch.albedo[ch.idx(x, y)] = Vec3(clamp01(max(col.x, 0.05)),
                                               clamp01(max(col.y, 0.05)),
                                               clamp01(max(col.z, 0.05)))

                // Roughness: matte slate face, rougher in the mortar joint and riven clefts.
                ch.roughness[ch.idx(x, y)] = clamp01(0.58 + 0.22 * joint + 0.10 * cleft + 0.06 * fine)

                // Height: flags stand proud, joints recessed; cleft micro-relief on the face.
                ch.height[ch.idx(x, y)] = clamp01(0.64 - 0.46 * joint + 0.14 * cleft - 0.06 * fine)
            }
        }
        ch.clearcoat = 0.10        // faint outdoor damp sheen on the slate face
        ch.deriveNormals(strength: 2.8)
        addMicroDetail(&ch, seed: seed ^ 0xDD, baseCells: 100, octaves: 2, strength: 0.55)  // riven grit
        return ch
    }

    /// Pebble / gravel ground — a dense bed of rounded stones, each its own warm grey/tan tone,
    /// packed tightly with dark shadow gaps between them. Built from a fine Voronoi scatter: the
    /// cell interior is a rounded pebble (bright, lifted), the cell boundary is the shaded gap
    /// (dark, recessed). Per-pebble colour jitter gives the macro speckle; the gap network and
    /// pebble crowns give strong height relief and spatially-varied roughness (smooth crowns,
    /// rough gaps). Outdoor-matte, no clearcoat.
    public static func pebble(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 149) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        // Pebble palette: warm grey-tan, ranging dark slate-grey → pale buff per stone.
        let dark = Vec3(0.30, 0.28, 0.25)
        let light = Vec3(0.66, 0.62, 0.56)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Dense pebble scatter — ~26 stones across the tile. grainPhase lands the tile
                // boundary at a stone centre, not the Voronoi lattice line, so the (already
                // toroidal) wrap column isn't a stone-boundary column — keeps TextureAudit's
                // seam ratio near 1 without altering the look. (See MaterialGenerator.grainPhase.)
                let cell = Noise.voronoiTiled(u + Self.grainPhase, v + Self.grainPhase,
                                              cells: 26, jitter: 1.0, seed: seed)
                // Crown: 1 at the pebble centre, falling to 0 at the gap between stones.
                let crown = smoothstep(0.0, 0.42, cell.f1)               // 0 centre … 1 gap (by f1)
                let pebbleHeight = 1.0 - crown                           // 1 centre … 0 gap
                // Gap shadow: where f2 - f1 is small we're in the dark crevice between stones.
                let gap = smoothstep(0.16, 0.02, cell.f2 - cell.f1)
                // Per-pebble colour + a little within-pebble grain.
                let tone = Double(cell.cellId & 0xFFFF) / 65535.0
                let grain = Noise.fbmTiled(u, v, baseCells: 60, octaves: 2, seed: seed ^ 0x57)

                var col = dark + (light - dark) * tone
                col += Vec3(0.05, 0.05, 0.05) * (grain - 0.5)            // wet/dry stone mottle
                col -= Vec3(0.16, 0.16, 0.15) * gap                     // shadowed crevices darken
                ch.albedo[ch.idx(x, y)] = Vec3(clamp01(max(col.x, 0.05)),
                                               clamp01(max(col.y, 0.05)),
                                               clamp01(max(col.z, 0.05)))

                // Roughness: stone crowns are smoother (rounded/polished), gaps rough with grit/dust.
                ch.roughness[ch.idx(x, y)] = clamp01(0.62 + 0.22 * gap + 0.10 * crown + 0.05 * grain)

                // Height: each pebble a rounded dome; deep recess in the gaps. Strong relief.
                ch.height[ch.idx(x, y)] = clamp01(0.30 + 0.55 * pebbleHeight - 0.20 * gap)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 3.2)
        addMicroDetail(&ch, seed: seed ^ 0xDD, baseCells: 120, octaves: 2, strength: 0.7)  // stone grit
        return ch
    }

    // MARK: - engineered quartz

    /// **World metres one engineered-quartz bake spans.**
    ///
    /// The aggregate is a real physical size, so — like tile, wood and terrazzo — the bake has to
    /// know how much surface it covers or it cannot draw a grain at any particular size. 1.0 m is
    /// the counter's own UV period (the millwork meshes register at a 1 m period), which is where
    /// this material lives; capping a wall/floor at the same number keeps a 5 mm grain 5 mm
    /// wherever the picker allows the material (`MaterialApplicability` lets `.stone` onto a
    /// floor, a wall and a backsplash), instead of stretching the bake over 2 m and rendering
    /// centimetre gravel.
    public static let quartzRunMeters: Double = 1.0

    /// The run a quartz bake gets on a given surface: our own period where the mesh takes its UVs
    /// from the material, and the mesh's period where it baked them earlier and would otherwise
    /// render a size we never chose. Same shape as `terrazzoRun(on:)` — one rule, two materials.
    public static func quartzRun(on tiling: SurfaceTiling) -> Double {
        tiling.isAdjustable ? Swift.min(tiling.metresPerRepeat, quartzRunMeters)
                            : tiling.metresPerRepeat
    }

    /// One crushed-grain population in an engineered quartz slab: how that mineral shifts the
    /// slab's base colour, how much of the aggregate it is, and the polish it takes.
    ///
    /// **Polish is per-mineral, and that is where quartz's sparkle comes from.** A translucent
    /// quartz crystal grinds to a harder face than the pigmented polyester binder around it, so a
    /// real slab's highlight breaks into flecks. Putting the whole roughness range on one
    /// low-frequency noise instead — which is what this material used to do — makes the same
    /// energy read as a gloss CLOUD, and a soft gloss cloud on a near-white surface is precisely
    /// how foam scatters.
    public struct QuartzGrain: Sendable, Equatable {
        /// Per-channel multiplier on the slab's base colour.
        public var tint: Vec3
        /// Fraction of grains that are this mineral. Need not sum to 1 — the draw normalises.
        public var share: Double
        /// Roughness this mineral takes at a polished finish.
        public var polish: Double

        public init(tint: Vec3, share: Double, polish: Double) {
            self.tint = tint; self.share = share; self.polish = polish
        }
    }

    /// **The aggregate of a white engineered quartz.** Mostly matrix-toned and warm-white filler,
    /// a quarter translucent quartz (the glassy grains), a grey mineral fraction, and a sparse
    /// dark pepper. The pepper is 3 % and it does most of the work: a field of near-white grains
    /// with nothing dark in it reads as plastic, which is the same lesson terrazzo's crusher-dust
    /// grade (`AggregateGrade.darkBias`) already records.
    public static let quartzGrains: [QuartzGrain] = [
        .init(tint: Vec3(1.005, 1.000, 0.992), share: 0.34, polish: 0.070),  // matrix-toned filler
        .init(tint: Vec3(1.045, 1.032, 1.010), share: 0.24, polish: 0.065),  // warm white
        .init(tint: Vec3(0.912, 0.918, 0.936), share: 0.24, polish: 0.032),  // translucent quartz
        .init(tint: Vec3(0.800, 0.800, 0.818), share: 0.16, polish: 0.080),  // grey mineral
        .init(tint: Vec3(0.575, 0.573, 0.598), share: 0.03, polish: 0.095),  // dark pepper
    ]

    /// Share-weighted mean polish of the aggregate — the datum the differential-relief term below
    /// measures each mineral against, derived from the table rather than typed beside it.
    public static let quartzMeanPolish: Double = {
        let total = quartzGrains.reduce(0) { $0 + $1.share }
        return quartzGrains.reduce(0) { $0 + $1.share * $1.polish } / Swift.max(total, 1e-9)
    }()

    /// Total of the shares, so the palette draw normalises instead of requiring the table to sum
    /// to exactly 1 by hand.
    public static let quartzGrainShareTotal: Double = quartzGrains.reduce(0) { $0 + $1.share }

    /// **The three crushed-aggregate sizes, in real millimetres**, and how much of the surface the
    /// two coarser ones claim. Crushed rock is graded — a slab with one grain size reads as
    /// sandpaper — so the mosaic is drawn coarsest-last over a fine bed: statement chips over mid
    /// fragments over the fine grain that fills everything between.
    /// The sizes are what separate this material from terrazzo, which is the OTHER crushed-stone
    /// mosaic in the library and the thing an over-graded quartz starts to look like: a terrazzo
    /// chip is 3–22 mm against a 0.6 m tile, i.e. a fragment you read one at a time, while quartz
    /// aggregate is a field you read as a texture. The fine bed sits at 5 mm because that is the
    /// smallest grain a 512 bake can carry over a 1 m counter (`drawableMM` ≈ 3.9 mm) — the real
    /// sub-millimetre population is simply not representable at this resolution.
    public static let quartzGrainMM = (fine: 5.0, mid: 10.0, chip: 20.0)
    public static let quartzMidShare = 0.28
    public static let quartzChipShare = 0.07

    /// How much height one unit of roughness difference is worth, i.e. how far the harder minerals
    /// stand proud of the softer binder. Sized so the relief is **present in the data and
    /// invisible in the render** — worst-case normal tilt 1.2°, gated in `QuartzCounterMaterialTests`.
    public static let quartzReliefPerRoughness = 0.6

    /// Engineered quartz counter (Silestone / Caesarstone style) — crushed quartz aggregate in a
    /// pigmented polymer binder, ground flat and polished.
    ///
    /// **It read as styrofoam** (Danny, 2026-08-20), and styrofoam is a specific combination:
    /// near-white, near-uniform, a faint isotropic mottle at centimetre-to-decimetre scale, no
    /// resolvable structure, and dimples. The old bake had all four, for three separate reasons.
    ///
    /// 1. **Wrong topology.** The aggregate was `1 − smoothstep(0.04, 0.12, f1)` over a 24-cell
    ///    Voronoi: one site per cell, masked down to a dot a few millimetres across. That is a
    ///    pepper of ISOLATED specks at well under 1 % coverage, on a visible lattice — and you can
    ///    see the rows and columns of it in the before frame. Engineered quartz is ~90 % aggregate
    ///    by weight: the grains TOUCH. Islands-in-a-binder is *terrazzo's* topology (which is why
    ///    terrazzo scatters chips with a separation test and this does not); a packed mosaic is
    ///    this material's, and a Voronoi tessellation — every texel belongs to some grain, cells
    ///    convex and straight-edged like crushed rock — is exactly that.
    /// 2. **All the energy at the wrong frequency.** With no grain to look at, the only spatial
    ///    structure left was a 3-cell fBm — a third of a metre per feature — driving tone (±3 %)
    ///    AND roughness (0.08…0.20). Decimetre mottle is the measured signature of the sponge-paint
    ///    defect; in the specular channel it is worse, because a gloss cloud is how foam scatters.
    ///    Everything here is now drawn at 5–30 mm, and the batch drift that remains is a whisper.
    /// 3. **A polished slab is GROUND FLAT.** The old height field embossed each speck (+0.20) and
    ///    cut vein grooves (−0.10) through `deriveNormals(strength: 2)` — tens of degrees of tilt,
    ///    i.e. a field of dimples on a white surface, which is the single strongest foam cue a
    ///    render can carry, and it is plainly visible in the before frame as a regular lattice of
    ///    lit bumps. Terrazzo already records this lesson ("a polished floor is GROUND FLAT… the
    ///    old half-height step would have embossed the whole surface into gravel"). The only relief
    ///    left is differential polish between minerals, at `strength: 0.4` over a ±0.02 field —
    ///    **measured worst-case tilt 1.2°** (`QuartzCounterMaterialTests.testTheSlabIsGroundFlat`),
    ///    which is present in the data and invisible in the render.
    ///
    /// **The veining is gone, deliberately.** A marble-look quartz is a different SKU; the white
    /// speckled stone this entry is named for has none, and the vein term was a second decimetre
    /// blotch source in both tone and roughness. If a veined quartz is ever wanted it is a params
    /// spine (like `TerrazzoParams`), not a wash smeared over every slab.
    ///
    /// Seamless: every layer is a toroidal `voronoiTiled` / `fbmTiled` lattice.
    public static func quartzCounter(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 88,
                                     baseColor: Vec3 = Vec3(0.880, 0.866, 0.845),
                                     runMeters: Double = MaterialGenerator.quartzRunMeters)
        -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .stone)
        let runMM = Swift.max(runMeters, 1e-3) * 1000
        /// Nyquist: a feature narrower than two texels cannot be drawn, only aliased. At the
        /// counter's 1 m period and a 512 bake this is 3.9 mm, so the fine grade sits just above
        /// the floor and nothing finer is attempted — the sub-millimetre population a real slab
        /// also has is simply not representable here, and pretending otherwise buys noise.
        let drawableMM = 2.0 * runMM / Double(size)
        /// Lattice period for a target grain size, floored at what this bake can resolve.
        func lattice(_ mm: Double) -> Int {
            Swift.max(1, Int((runMM / Swift.max(mm, drawableMM)).rounded()))
        }
        let fineCells = lattice(quartzGrainMM.fine)
        let midCells = lattice(quartzGrainMM.mid)
        let chipCells = lattice(quartzGrainMM.chip)

        /// One grain's look, from its Voronoi cell id: a weighted palette draw, a per-grain value
        /// jitter (no two fragments of the same mineral are quite the same), and the polish that
        /// draw implies.
        func grain(_ id: UInt64, spread: Double) -> (color: Vec3, polish: Double) {
            let h = Noise.mix(id)
            var pick = Noise.unit(h) * quartzGrainShareTotal
            var i = 0
            while i < quartzGrains.count - 1 && pick >= quartzGrains[i].share {
                pick -= quartzGrains[i].share
                i += 1
            }
            let g = quartzGrains[i]
            let jitter = 1 + spread * (Noise.unit(Noise.mix(h ^ 0x51)) - 0.5)
            return (baseColor * g.tint * jitter, g.polish)
        }

        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size), v = (Double(y) + 0.5) / Double(size)

                // Three nested mosaics, fine bed first and the coarse fragments laid over it. Each
                // layer OVERWRITES the one beneath where it is present, so the result is a packed
                // tessellation with no binder gaps anywhere — which is what ~90 % aggregate means.
                var g = grain(Noise.voronoiTiled(u, v, cells: fineCells, jitter: 1.0,
                                                 seed: seed).cellId, spread: 0.10)
                let mid = Noise.voronoiTiled(u, v, cells: midCells, jitter: 1.0, seed: seed ^ 0x2B)
                if Noise.unit(Noise.mix(mid.cellId ^ 0xA7)) < quartzMidShare {
                    g = grain(mid.cellId, spread: 0.07)
                }
                let chip = Noise.voronoiTiled(u, v, cells: chipCells, jitter: 1.0, seed: seed ^ 0x4D)
                if Noise.unit(Noise.mix(chip.cellId ^ 0xC3)) < quartzChipShare {
                    g = grain(chip.cellId, spread: 0.05)
                }

                // **Batch drift, and it is a whisper on purpose.** A slab does carry a slow cloud,
                // but at 30 cm any visible amount of it reads as blotch rather than stone. ±0.8 %
                // on a near-white is under 2 sRGB code values once `MaterialBaker` encodes the
                // albedo, i.e. below what survives the atlas as a mottle, while still keeping the
                // field off mathematically flat.
                let drift = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: seed ^ 0x44) - 0.5
                ch.albedo[ch.idx(x, y)] = clampBand(g.color * (1 + 0.016 * drift))

                // Polish per grain, plus a grain-scale dither so two neighbouring fragments of the
                // same mineral still catch the light slightly differently.
                let dither = Noise.fbmTiled(u, v, baseCells: fineCells, octaves: 1,
                                            seed: seed ^ 0x6E) - 0.5
                ch.roughness[ch.idx(x, y)] = clamp01(g.polish + 0.012 * dither)

                // Differential polish: the harder minerals stand a few microns proud of the softer
                // binder. This is the ONLY relief a ground-and-polished slab has.
                ch.height[ch.idx(x, y)] =
                    clamp01(0.5 + (quartzMeanPolish - g.polish) * quartzReliefPerRoughness)
            }
        }
        ch.clearcoat = 0.30                          // polished stone, the same lobe marble gets
        ch.deriveNormals(strength: 0.4)              // ≈0.5° maximum tilt — see the note above
        return ch
    }

    /// Butcher block counter — **hard-maple EDGE-GRAIN strips**, delegated to the shared
    /// `woodGrain` engine like every other plank wood in the library.
    ///
    /// **It used to be bespoke, and it was broken two ways.** The ring term was
    /// `pow(0.5 − 0.5·cos((ringBase · 12 + phase) · 2π), 3)` where `ringBase` is an fBm in
    /// `[0, 1]`: multiplying a smooth noise field by 12 inside a cosine makes the cosine sweep
    /// twelve full cycles across the noise's range, so a texel-to-texel wobble of a few per cent
    /// jumps most of a cycle. That converts smooth noise into a rapidly oscillating field — the
    /// swatch read as **particleboard grit / carved damask**, not wood, and no colour or strength
    /// tuning could fix it because the frequency was wrong, not the amplitude. (Same failure
    /// shape as the paint diagnosis: [[daydream-wall-texture-quality-bar]].)
    ///
    /// **And the anatomy was wrong for a counter.** It modelled END grain (concentric rings, the
    /// chopping-block look). A butcher-block COUNTERTOP is edge-grain: narrow strips laminated
    /// face to face, each showing long, nearly-straight grain running down the strip with a glue
    /// line between. That is exactly what `woodGrain` already draws — long latewood lines along
    /// V, per-board tone/phase jitter, and a recessed side-seam groove per strip — so this is now
    /// eight narrow boards of maple rather than forty lines of its own geometry
    /// ([[feedback-unify-sibling-entities]], [[feedback_code_reuse]]). `patternCells` /
    /// `patternJitter` come from `woodGrain` too, so the per-strip de-repeat is declared once.
    public static func butcherBlock(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 99) -> MaterialChannels {
        // 8 strips over the counter's 1.0 m tile ⇒ ~125 mm laminations, the standard width.
        // 8 divides 512 evenly, so both wrap columns land inside a glue zone (seam-safe).
        // Tight rings and a nearly-straight bow (`WoodRecipe.hardMaple`): edge grain shows the
        // rings almost edge-on, which is why a butcher block reads as fine parallel lines rather
        // than oak's cathedral arches. Continuous laminations — glue lines run the full length,
        // and a butcher block has no staggered butt joints across them.
        return woodGrain(size: size, planks: butcherBlockStrips, seed: seed,
                         recipe: .hardMaple, runMeters: butcherBlockRunMeters,
                         category: .wood, endJoints: false)
    }

    /// Laminations per butcher-block tile. Must divide the bake size evenly (seam safety).
    public static let butcherBlockStrips = 8

    /// World metres one butcher-block repeat spans — the counter tile it is quoted at. Ring
    /// spacing is physical (`WoodRecipe.ringsPerMeter`), so the bake has to know its own size;
    /// 8 strips over 1.0 m is the standard ~125 mm lamination.
    public static let butcherBlockRunMeters = 1.0
}
