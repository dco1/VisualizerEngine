import simd
import Foundation

extension MaterialGenerator {
    /// Brushed stainless steel — the **metal + anisotropy** path (§5/§3). Metals carry
    /// no diffuse; the base color is the spec tint. The look lives in fine directional
    /// brush streaks: roughness varies *across* the grain and runs *along* it, with the
    /// grain tangent fed to the engine's anisotropy lobe so the highlight stretches across
    /// the brush (tell #7). Grain is horizontal here.
    ///
    /// **The albedo carries NO streak, and that is the 2026-08-11 fix.** Danny reported the
    /// kitchen refrigerator reading as **tan corduroy** (PHOTOREALISM #10's "flagged
    /// separately" list). Both halves of that were authored here:
    ///
    /// * **Corduroy.** `albedo = base * (0.88 + 0.20 · streak)` put a ±10 % TONE modulation
    ///   on the brush pattern. Measured by `MaterialToneVarianceAuditTests`: albedo luma
    ///   CV 2.36 % total, **98 % of it grid-structured**, half-scale **16 cm** — i.e. a
    ///   directional tonal banding at decimetre scale, which at millwork's 1.0 m tile is a
    ///   ~12.5 cm along-grain band crossed by ~1 cm ribs. That is a corduroy weave, drawn in
    ///   tone, on a surface whose real grain is sub-millimetre. **For a metal it is also
    ///   wrong in principle**: albedo IS F0, so modulating it modulates *reflectance colour*
    ///   — the streak was literally painting stripes of differently-coloured steel. This is
    ///   the same correction paint got in S2.2 (variance belongs in roughness, not tone), and
    ///   it applies harder here because a metal has no diffuse for tone to live in. Metals are
    ///   `TextureAudit` tone-EXEMPT, so nothing was requiring it.
    /// * **Tan.** `base` was `(0.66, 0.64, 0.60)` — R−B = +0.06, a warm beige. Real 304
    ///   stainless is near-neutral and faintly COOL. Now `(0.58, 0.585, 0.59)`.
    ///
    /// What replaces it is what brushed steel actually is: the same brush pattern in
    /// ROUGHNESS and HEIGHT (untouched, and slightly deepened in roughness to carry the look
    /// the tone was doing), the grain tangent, and the anisotropic lobe.
    ///
    /// `legacyTonalBrushStreakForTest` reproduces the pre-fix bake exactly — same binary, same
    /// process, so the A/B is a measurement rather than a comparison against a remembered
    /// frame. Production never sets it.
    public nonisolated(unsafe) static var legacyTonalBrushStreakForTest = false

    /// Amplitude of the brush streak in ROUGHNESS — the one lever that decides whether this
    /// material reads as satin or as corduroy, and the reason is the PERIOD it has to live at.
    ///
    /// The across-grain streak is 96 cells per tile; millwork bakes at a 1.0 m tile, so the rib
    /// period on a fridge door is ~10 mm. Real brushed stainless is sub-millimetre, and a 512²
    /// bake at 1 m cannot express that at all (one texel is 1.95 mm). So the pattern is stuck
    /// an order of magnitude too coarse, and the only honest lever left is how LOUD it is.
    ///
    /// Left at 0.22. It was raised to 0.30 "to carry what the tone was doing", then measured:
    /// across a 7.5× sweep (0.30 → 0.04) the rendered panel is indistinguishable. Roughness is
    /// not the carrier here; `brushStreakHeightAmp` below is.
    ///
    /// **0.10 since DH-0965 (2026-09-30).** The note above was measured on a bake whose streak
    /// repeated inside the tile and whose finer octaves were past Nyquist; on the band-limited
    /// `brushedMetal` recipe the gloss streak IS a visible centimetre band, so it is held to a
    /// faint drift (0.10 and 0.04 were indistinguishable on the Sunset range sheet).
    public nonisolated(unsafe) static var brushStreakRoughnessAmp: Double = 0.10

    /// Amplitude of the brush streak in HEIGHT — `deriveNormals` turns this into the normal
    /// map, and **this is what drew the corduroy.**
    ///
    /// **0.5 was a units error wearing a taste costume.** A brush groove on real stainless is
    /// microns deep at a ~0.1 mm pitch. Here the pitch is forced to ~10 mm by millwork's 1.0 m
    /// tile (and by a 512² bake, where one texel is 1.95 mm) — roughly 100× too coarse. Holding
    /// the DEPTH fixed while the PITCH grows 100× multiplies the surface slope by 100, and what
    /// that renders is not a brushed finish at the wrong scale, it is **corrugation**. At the
    /// kitchen hero the refrigerator read as ribbed cardboard: PHOTOREALISM #10's "tan corduroy".
    ///
    /// **0.20 ships**, chosen by looking at a four-point sweep at that rig
    /// (`steel-corduroy-{before,h20,h8,h3}`): 0.5 corrugates, **0.20 keeps a fine directional
    /// grain that reads as brushed satin**, 0.08 is nearly gone and 0.03 is a smooth panel. So
    /// the brushed read survives and the ribs do not.
    ///
    /// **0 since DH-0965 (2026-09-30): the brushed family bakes NO relief.** At ~2 mm a texel any
    /// relief is a centimetre stripe — 0.20 on the rebuilt recipe drew the horizontal banding
    /// Danny called out on the Sunset range ("still needs refining on the horizontal"), and every
    /// arm at ≤ 0.05 read as flat satin beside maps-off stainless. The brush a person sees is the
    /// detail-band hairline (`brushHairlineStrength`) plus the anisotropic lobe.
    public nonisolated(unsafe) static var brushStreakHeightAmp: Double = 0

    public static func brushedSteel(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 6,
                                    base baseIn: Vec3 = Vec3(0.58, 0.585, 0.59)) -> MaterialChannels {
        let legacy = legacyTonalBrushStreakForTest
        guard legacy else {
            return brushedMetal(size: size, seed: seed, base: baseIn, roughness: 0.28,
                                acrossCells: 96, clearcoat: 0, microSeed: seed ^ 0xB2)
        }
        // The pre-2026-08-11 bake, frozen for the corduroy A/B — including the periodic
        // `v · 24` stripe the shared recipe below no longer draws. Never shipped.
        let base = Vec3(0.66, 0.64, 0.60)
        var ch = MaterialChannels(size: size, category: .metal)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let streak = Noise.fbmTiled(u * 2, v * 24, baseCells: 4, octaves: 3, seed: seed)
                let fine = Noise.fbmTiled(u * 1, v * 64, baseCells: 6, octaves: 2, seed: seed ^ 0x55)
                ch.albedo[ch.idx(x, y)] = clampBand(base * (0.88 + 0.20 * streak))
                ch.roughness[ch.idx(x, y)] = clamp01(0.28 + (streak - 0.5) * 0.22 + (fine - 0.5) * 0.10)
                ch.height[ch.idx(x, y)] = clamp01(0.5 + (streak - 0.5) * 0.5)
            }
        }
        ch.grainTangent = [Vec2](repeating: Vec2(1, 0), count: size * size)
        ch.deriveNormals(strength: 2)
        addMicroDetail(&ch, seed: seed ^ 0xB2, baseCells: 100, strength: 0.30)
        return ch
    }

    /// **The brushed-metal recipe — ONE for the whole family** (Brushed Steel / Stainless,
    /// Aluminum, Brushed Nickel). Each used to carry its own, and the two copies that were not
    /// brushed steel's had drifted into every defect brushed steel had already been cured of
    /// (Daydream DH-0965 — a range finished in Aluminum read as corrugated sheet):
    ///
    /// * **Stretched on a non-square lattice, never a scaled coordinate.** `fbmTiled(u · 40, …)`
    ///   tiles one strip forty times: an exactly periodic 25 mm rib on a 1 m millwork tile.
    ///   `Noise.fbmTiledAniso` puts `alongCells × acrossCells` distinct cells over the tile.
    /// * **Streaked ALONG the grain it declares.** Features are long in U (few cells along,
    ///   many across) and the tangent is +U — Aluminum and Nickel had it the other way round,
    ///   so the anisotropic lobe stretched the highlight across the streaks it was drawn over.
    /// * **Flat albedo.** A metal's albedo IS its F0; the brush is roughness and relief
    ///   (the 2026-08-11 "tan corduroy" correction, above).
    /// * **Band-limited.** No octave finer than two texels per cell — past that the lattice
    ///   is per-texel noise, which the normal derivation turns into sparkle.
    ///
    /// The streak amplitudes are brushed steel's measured ones (`brushStreakRoughnessAmp`,
    /// `brushStreakHeightAmp`); a member differs only in its tint, satin level, lacquer and how
    /// fine its brush is (`acrossCells` per tile).
    static func brushedMetal(size: Int, seed: UInt64, base: Vec3, roughness: Double,
                             acrossCells: Int, clearcoat: Double, microSeed: UInt64) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        let along = 8
        let limit = max(along, size / 2)           // two texels per cell, even on the first octave
        let across = min(acrossCells, limit)
        let streakOctaves = Noise.bandLimitedOctaves(3, baseCells: across, size: size)
        let fineAcross = min(across * 4, limit)
        let fineOctaves = Noise.bandLimitedOctaves(2, baseCells: fineAcross, size: size)
        let flat = clampBand(base)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let streak = Noise.fbmTiledAniso(u, v, cellsX: along, cellsY: across,
                                                 octaves: streakOctaves, seed: seed)
                let fine = Noise.fbmTiledAniso(u, v, cellsX: along, cellsY: fineAcross,
                                               octaves: fineOctaves, seed: seed ^ 0x55)
                ch.albedo[ch.idx(x, y)] = flat
                ch.roughness[ch.idx(x, y)] = clamp01(
                    roughness + (streak - 0.5) * brushStreakRoughnessAmp + (fine - 0.5) * 0.10)
                ch.height[ch.idx(x, y)] = clamp01(0.5 + (streak - 0.5) * brushStreakHeightAmp)
            }
        }
        ch.grainTangent = [Vec2](repeating: Vec2(1, 0), count: size * size)   // brush runs along U
        ch.clearcoat = clearcoat
        ch.deriveNormals(strength: 2)
        // The HAIRLINE — the brush a person actually sees, drawn where it can exist. A real
        // brushed line is ~0.1–0.5 mm; the base bake is ~2 mm a texel on a 1 m tile, so there it
        // can only be a centimetre stripe (the banding). The detail band tiles
        // `detailNormalUVScale` (8×) finer — ~0.24 mm a texel — and a metal is the one surface
        // whose specular shows it (the detail band is inert on dielectrics). Long along U with
        // the grain tangent, a few along-cells so each line runs centimetres; one octave, held
        // to two texels a cell.
        let hairAlong = 4
        let hairAspect = max(1, min(64, (size / 2) / hairAlong))
        addMicroDetail(&ch, seed: microSeed, baseCells: hairAlong, octaves: 1,
                       strength: brushHairlineStrength, grainAspect: hairAspect, grainAlongU: true)
        return ch
    }

    /// Strength of the brushed family's hairline detail normal (`brushedMetal`). Chosen off the
    /// real-Metal Sunset range sheet (`HouseRenderBridgeGPUTests_BrushedMetalSheet`, DH-0965).
    /// **0.20.** The hairline is sub-pixel at a room camera, so a stronger one ALIASES into a
    /// fine horizontal hatch: 0.45 read clean under the older still-lane denoiser but hatched
    /// once the still started preserving surface grain (round 2 of the sheet: both 0.45 arms
    /// hatched with or without the gloss streak, 0.20 and 0 did not). 0.20 reads as smooth satin
    /// beside maps-off stainless and still leaves the detail band live for close range.
    public nonisolated(unsafe) static var brushHairlineStrength: Double = 0.20

    /// Polished brass — a warm **isotropic metal** (no diffuse; the base is the spec
    /// tint) with broad fBm tone drift and the odd darker **patina** bloom. No grain
    /// (it isn't brushed), so it legitimately carries none.
    public static func brass(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 8,
                             base: Vec3 = Vec3(0.91, 0.71, 0.34),
                             patina: Vec3 = Vec3(0.40, 0.52, 0.45)) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                let tone = 0.90 + 0.16 * Noise.fbmTiled(u, v, baseCells: 3, octaves: 4, seed: seed)
                let bloom = pow(Noise.fbmTiled(u, v, baseCells: 4, octaves: 4, seed: seed ^ 0x3C), 4) * 0.6
                ch.albedo[ch.idx(x, y)] = clampBand(mix(base * tone, patina, bloom))
                ch.roughness[ch.idx(x, y)] = clamp01(0.18 + bloom * 0.4
                    + (Noise.fbmTiled(u, v, baseCells: 26, octaves: 2, seed: seed ^ 0x4D) - 0.5) * 0.05)
                ch.height[ch.idx(x, y)] = clamp01(0.5 - bloom * 0.15)
            }
        }
        ch.deriveNormals(strength: 2)
        return ch
    }

    /// Polished copper — the warm-metal family at a redder base, with a **verdigris**
    /// (blue-green) patina bloom in the recesses. Common for fixtures, range hoods, and
    /// accent hardware. Shares the brass tone-drift + patina recipe (§5 metals list).
    public static func copper(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 14) -> MaterialChannels {
        brass(size: size, seed: seed,
              base: Vec3(0.95, 0.64, 0.54),        // pink-orange copper
              patina: Vec3(0.26, 0.56, 0.52))      // verdigris blue-green
    }

    /// Polished bronze — the warm-metal family at a darker brown-gold base with a dusky
    /// brown-green tarnish. Reads heavier/older than brass; common for hardware and trim.
    public static func bronze(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 16) -> MaterialChannels {
        brass(size: size, seed: seed,
              base: Vec3(0.64, 0.47, 0.27),        // dark brown-gold
              patina: Vec3(0.30, 0.36, 0.28))      // dusky brown-green tarnish
    }

    /// **Corten (weathering steel)** — the rust-patina metal a modern garden planter / edging
    /// wears (DH-0124). A `.metal` (so it renders metallic, and the tone tell is correctly
    /// lifted — a metal's albedo is its reflectance), but a MATTE one: the whole look is the
    /// oxide, so roughness sits high (0.55–0.85) and drifts spatially. The weathering is
    /// authored VERTICALLY — rain washes the oxide down the face in streaks — so the drift is
    /// long down V (`fbmTiledAniso`, 96 × 4 cells), with darker rain-runs and the odd
    /// brighter fresh-oxide bloom. Isotropic (no grain tangent): the streaks are tone, not a
    /// brushed grain the engine should stretch a highlight along.
    public static func corten(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 211,
                              base: Vec3 = Vec3(0.44, 0.23, 0.13),      // warm rust orange-brown
                              bloom: Vec3 = Vec3(0.58, 0.36, 0.21),     // lighter fresh-oxide flush
                              runoff: Vec3 = Vec3(0.24, 0.13, 0.09)     // dark rain-streak stain
    ) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        let runOctaves = Noise.bandLimitedOctaves(3, baseCells: 96, size: size)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Broad blotchy patina + vertical rain-runs + fine pit mottle.
                let patch  = Noise.fbmTiled(u, v, baseCells: 3, octaves: 4, seed: seed)
                // Runs are LONG in V (world-vertical on every wall-facing surface): many cells
                // across U, few down V, on a non-square lattice. This was `fbmTiled(u, v · 24)` —
                // features 24× longer in U than V (horizontal bands, not runs) and one strip
                // tiled 24 times down the face (DH-0965).
                let streak = Noise.fbmTiledAniso(u, v, cellsX: 96, cellsY: 4,
                                                 octaves: runOctaves, seed: seed ^ 0x5C)
                let grit   = Noise.fbmTiled(u, v, baseCells: 30, octaves: 2, seed: seed ^ 0x2B)
                let run    = pow(streak, 3) * 0.6                        // dark vertical runoff, sparse
                let flush  = pow(patch, 4) * 0.5                          // bright fresh-oxide bloom
                var c = mix(base * (0.90 + 0.18 * patch), bloom, flush)
                c = mix(c, runoff, run)
                ch.albedo[ch.idx(x, y)] = clampBand(c)
                // Matte oxide: rough everywhere, roughest in the runoff, a touch smoother at a bloom.
                ch.roughness[ch.idx(x, y)] = clamp01(0.60 + 0.20 * run + 0.10 * grit
                    - 0.08 * flush + (patch - 0.5) * 0.10)
                ch.height[ch.idx(x, y)] = clamp01(0.5 + (grit - 0.5) * 0.5 - run * 0.15)
            }
        }
        ch.clearcoat = 0.0                          // no lacquer — bare weathered oxide
        ch.deriveNormals(strength: 2.5)
        addMicroDetail(&ch, seed: seed ^ 0xF3, baseCells: 84, octaves: 2, strength: 0.85)   // oxide tooth
        return ch
    }

    /// Brushed nickel — warm silver fixture finish: a finer brush than steel under a thin
    /// lacquer. A member of the `brushedMetal` family; it used to carry its own recipe, which
    /// streaked ACROSS its grain tangent as an exactly periodic 15.6 mm rib painted into the
    /// albedo (DH-0965).
    public static func brushedNickel(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 113) -> MaterialChannels {
        brushedMetal(size: size, seed: seed, base: Vec3(0.72, 0.70, 0.67), roughness: 0.22,
                     acrossCells: 160, clearcoat: 0.20, microSeed: seed ^ 0xBD)
    }

    /// The colourway a bare `matteBlack()` bakes — the near-neutral powder-coat black, luma 0.10
    /// with the historical faint-warm tint (×0.98 green, ×0.96 blue). Declared once so the
    /// registry's `matte-black` default and the generator's default `color:` cannot drift apart.
    // DH-1397: was luma 0.10; black powder coat / black paint measures 0.03-0.06.
    public static let matteBlackDefaultColor = Vec3(0.062, 0.060, 0.058)

    /// Matte black — powder-coat / anodized finish. Deep absorptive, micro-textured.
    ///
    /// `color` is a **colourway** (DH-0470): it drives the albedo ONLY. A powder coat's life is
    /// its orange-peel tooth and its roughness field, and those are the SAME finish whatever the
    /// pigment — a white, bronze or sage powder-coat is cast and cured exactly like the black one.
    /// So the roughness field and the cast micro-relief below are identical for every colour, and
    /// only the albedo follows `color`. Bare `matteBlack()` bakes the historical black bit-for-bit
    /// (`matteBlackDefaultColor` is the old `Vec3(luma, luma*0.98, luma*0.96)` mean).
    public static func matteBlack(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 117,
                                  color: Vec3 = MaterialGenerator.matteBlackDefaultColor) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        let sh = seed
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let micro = Noise.fbmTiled(u, v, baseCells: 6, octaves: 4, seed: sh)
                let coat  = Noise.fbmTiled(u, v, baseCells: 15, octaves: 3, seed: sh ^ 0xAA)   // 15 distinct cells, not 5 tiled 3 × 3 (DH-0965)
                // Powder-coated matte black is one of the most tonally UNIFORM finishes
                // there is — its life is entirely the orange-peel tooth below, not blotch.
                //
                // This used to be `0.05 + micro*0.06 + coat*0.04`, i.e. luma swinging
                // 0.05…0.15 — a **3× tone range at decimetre scale on a near-black
                // surface** (micro is 6 cells over a 1.5 m tile ≈ 25 cm, coat ≈ 10 cm).
                // The S2.1 library audit ranked it the worst offender among architectural
                // finishes: 8.51 % coarse albedo CV, 80 % of variance low-frequency,
                // grid-structure score 16 (i.e. pure unstructured mottle). That is the
                // "a lot of the parts look like black mold" report (Danny, 2026-08-05),
                // and matte-black is on every stair railing, baluster, newel, lamp base
                // and piece of hardware in the house.
                //
                // Same treatment as `paint`: the variance moves OUT of tone and stays in
                // roughness (untouched below, ±0.22 micro / ±0.10 coat) and in the relief
                // normal. The residual ±0.15 % tone whisper keeps the bake off a
                // mathematically dead flat fill without being visible (well under one 8-bit
                // code value at the black's luminance). It is applied MULTIPLICATIVELY so it
                // scales with the chosen colourway rather than being a fixed black offset —
                // a white powder-coat gets the same ±1.5 % relative whisper, not a ±0.0015
                // one that would vanish against a 0.85 base. At the default black this is the
                // historical `0.10 + micro*0.002 + coat*0.001`.
                let toneVar = (micro - 0.5) * 0.002 + (coat - 0.5) * 0.001
                ch.albedo[ch.idx(x, y)] = clampBand(color * (1.0 + toneVar / 0.10))
                ch.roughness[ch.idx(x, y)] = clamp01(0.68 + (micro - 0.5) * 0.22
                    + (coat - 0.5) * 0.10)
                ch.height[ch.idx(x, y)] = clamp01(0.50 + (micro - 0.5) * 0.10)
            }
        }
        ch.deriveNormals(strength: 2)
        // Powder-coat cast texture — the fine orange-peel tooth that keeps matte black from
        // reading as glossy injection-molded plastic. This is the clearest plastic-tell fix.
        addMicroDetail(&ch, seed: seed ^ 0xBE, baseCells: 70, strength: 0.60)
        return ch
    }

    /// Chrome / polished aluminum — mirror-bright metal, very low roughness,
    /// high reflectance. No grain (isotropic). Warm silver with micro pitting.
    public static func chrome(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 121) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        let sh = seed
        // DH-1397: F0 of chromium (n≈3.2, k≈3.3) is 0.55-0.66; the old 0.90 was aluminium/silver.
        let base = Vec3(0.67, 0.66, 0.65)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let micro = Noise.fbmTiled(u, v, baseCells: 4, octaves: 4, seed: sh)
                let pit = Noise.fbmTiled(u, v, baseCells: 22, octaves: 3, seed: sh ^ 0xB1)
                let pitting = pow(max(0, 1 - pit * 1.4), 6) * 0.04
                let tone = 0.94 + 0.12 * micro
                ch.albedo[ch.idx(x, y)] = clampBand(base * tone)
                // Enough roughness variation to pass TextureAudit while keeping chrome
                // appearance near-mirror: base 0.06, micro variation ±0.06.
                ch.roughness[ch.idx(x, y)] = clamp01(0.06 + micro * 0.12 + pitting)
                ch.height[ch.idx(x, y)] = clamp01(0.55 - pitting * 3)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 1)
        return ch
    }

    /// **Mirror plate** — silvered glass. The flattest, brightest surface in the library: a
    /// near-perfect specular reflector, which is the whole point — the engine's screen-space
    /// reflections (`ssrIntensity`, on by default) are what turn this panel into a mirror, and SSR
    /// only reads as a mirror at very low roughness. Chrome is close but not the same material: a
    /// chrome tap is a warm-tinted metal with visible micro-pitting, a mirror is neutral and
    /// flawless, so a mirror wearing `chrome` reads as brushed steel.
    ///
    /// Category `.mirror` — the case `MaterialApplicability` already declared (wall-only) and
    /// nothing had ever filled. `TextureAudit` treats `.mirror` like metal/glass, so the
    /// flat-roughness tell that would fail a dielectric doesn't apply: a mirror IS flat, and
    /// varying its roughness to satisfy a generic audit would literally frost it.
    ///
    /// The faint large-scale roughness/height wobble is deliberate and tiny (±0.008): real plate
    /// glass has a slow waviness from the float process, and it keeps the reflection from being
    /// mathematically perfect (which reads as CGI) without softening it.
    public static func mirrorGlass(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 733) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .mirror)
        let sh = seed
        // Silvered glass is very slightly cool and very bright — a real mirror returns ~95%.
        let base = Vec3(0.95, 0.96, 0.97)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                // Float-glass waviness: very low frequency, very low amplitude.
                let wave = Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: sh) - 0.5
                ch.albedo[ch.idx(x, y)] = clampBand(base * (1.0 + wave * 0.012))
                ch.roughness[ch.idx(x, y)] = clamp01(0.018 + wave * 0.016)
                ch.height[ch.idx(x, y)] = clamp01(0.5 + wave * 0.02)
            }
        }
        ch.clearcoat = 0.0
        ch.deriveNormals(strength: 0.15)      // barely any normal relief — a mirror is FLAT
        return ch
    }

    /// Brushed aluminium — the brightest, slightly more satin member of the `brushedMetal`
    /// family, under a thin anodised coat. Its own recipe used to be
    /// `fbmTiled(u · 40, v · 3, baseCells: 8)`: one 8-cell strip tiled forty times, a 25 mm rib
    /// streaked ACROSS its grain tangent and painted ±15 % into the albedo — the corrugated range
    /// front of DH-0965.
    public static func aluminum(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 131) -> MaterialChannels {
        brushedMetal(size: size, seed: seed, base: Vec3(0.77, 0.77, 0.76), roughness: 0.32,
                     acrossCells: 128, clearcoat: 0.10, microSeed: seed ^ 0xC0)
    }
}
