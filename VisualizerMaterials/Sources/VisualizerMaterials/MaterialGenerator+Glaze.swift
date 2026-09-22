import simd

// **Glazed baked clay** — the material a ceramicist produces, with the colour and the finish
// picked by the user (`GlazeParams`). The whole family lives in this file, next to the
// `GlazeFinish` ladder that supplies its physics.
//
// It has three siblings in the library and is none of them:
//   • `ceramicTile` is a tiled floor/wall surface WITH a grout grid — unit masonry, not a pot.
//   • `ceramic` is glossy white SANITARYWARE, slip-cast in a mould: one fixed colour, one fixed
//     gloss, and deliberately no evidence of a hand anywhere on it.
//   • `terracotta` is unglazed garden earthenware in one fixed orange-red.
// None of the three takes a colour, none takes a finish, and none of them was THROWN. This one
// is: it carries the wheel's rings, an unevenly poured glaze, and a clay body that shows where
// the coat runs thin — the three things that separate studio pottery from a moulded planter.
extension MaterialGenerator {

    // ── TASTE DIALS ───────────────────────────────────────────────────────────────
    // Flagged for Danny. The generator derives everything else from these plus the picked
    // `GlazeParams`, so there is one place to change what the clay under the glaze is made of.

    /// **The clay body under the glaze** — a buff/toast stoneware, the commonest studio clay.
    /// This is what shows through wherever the coat is thin (`GlazeFinish.bodyBreak`), and it is
    /// what makes a hand-glazed pot read as two materials rather than one flat colour. Pull it
    /// toward `(0.52, 0.28, 0.19)` for a red earthenware body, or toward `(0.78, 0.75, 0.70)`
    /// for a white porcelain one.
    public static let glazeClayBody = Vec3(0.52, 0.42, 0.32)

    /// **The iron speck** the body blooms up through a glaze. A dark warm brown, not black —
    /// black is not a physically plausible dielectric albedo (see `clampBand`).
    public static let glazeIronSpeck = Vec3(0.10, 0.062, 0.040)

    /// **The stain in a crazed line.** A crackle glaze's network only becomes visible once the
    /// hairlines have taken up tannin from years of use; a fresh craze is invisible, so drawing
    /// the aged one is drawing the thing anybody would recognise.
    public static let glazeCrazeStain = Vec3(0.34, 0.29, 0.24)

    /// **Iron-speck size and spacing**, in metres. A speck is DRAWN AT A CELL CENTRE — see the
    /// generator's `fleck` term for the expression that took two tries to get right.
    ///
    /// 4 mm is coarser than life and it is the smallest honest speck this bake can hold. Real
    /// iron speckle in a stoneware body is 0.5–2 mm; at the 1 m vessel run a 512² texel is
    /// 1.95 mm, so anything under ~4 mm is below the two-texel Nyquist floor and cannot be drawn
    /// at all — the same wall `MaterialGenerator.paint`'s orange peel hit before its repeat came
    /// down. Drawing it anyway would produce a sub-texel shimmer, not a finer speck.
    public static let glazeSpeckSize = 0.0044
    public static let glazeSpeckSpacing = 0.0156

    /// **Crazing cell size and hairline width**, in metres. A real crazed glaze is a mesh of
    /// ~1 cm cells separated by lines you can barely see; the first cut drew 3.3 cm cells with
    /// 8 mm lines and the pot came out looking like giraffe skin.
    ///
    /// The width then went the other way and hit the SAME two-texel floor `glazeSpeckSize` is
    /// pinned to — at 2.8 mm the line is 1.4 texels and the network breaks into disconnected
    /// dots (measured: its largest connected component held 0.4 % of the marked texels, against
    /// the > 55 % a network must hold). 4 mm is the thinnest line this bake can draw as a LINE.
    /// Real crazing is finer than this and cannot be, for the same Nyquist reason.
    public static let glazeCrazeCell = 0.0172
    public static let glazeCrazeWidth = 0.0040

    /// **Throwing-ring pitch, in metres** — the spiral rib a potter's fingers leave climbing the
    /// wall of a pot. 10 mm is a fast confident pull; 4–6 mm is a slow one.
    ///
    /// The number is physical, and it is *representable*, which is the part that took measuring.
    /// Every vessel mesh in the app bakes its UVs at a 1 m period (`ElementUVScale.plantPot` /
    /// `.accessory`, both `prebaked(1.0)`), so the 512² bake spans 1 m and one texel is 1.95 mm.
    /// At 10 mm a ring is **5.1 texels**, comfortably clear of the two-texel Nyquist floor. At the
    /// 4 mm end it would be two texels and could not be drawn at all — the same trap the paint
    /// tile's orange peel was in before its repeat came down (see `MaterialGenerator.paint`).
    public static let glazeThrowingRingPitch = 0.010
    /// The bake's nominal world span, in metres — see `glazeThrowingRingPitch`. Not declared as a
    /// `runMeters(on:)` opinion, deliberately: this material is only ever applied to a VESSEL or
    /// small accessory (`MaterialApplicability` refuses it on a room surface), and every one of
    /// those meshes is already prebaked at exactly this period, so an opinion could only ever
    /// restate what the surface already says — or, on a surface that changed later, silently
    /// retile it.
    public static let glazeNominalRun = 1.0

    /// **Throwing-ring normal amplitude**, in height-field units, at `throwRelief == 1`.
    ///
    /// `deriveNormals(strength:)` takes a two-texel central difference, so a sine of amplitude
    /// `A` and period `P` texels produces a tangent slope of `strength · 2A · 2π / P`. At
    /// `P = 5.1` and `strength = glazeNormalStrength` the value below puts the rings at ≈ 6.1° at
    /// full `throwRelief`, and ≈ 2.7° at the satin default a pot ships with.
    ///
    /// It is bounded on both sides, and both bounds were found on the real Metal frame:
    ///
    /// - **Above**, at the amplitude a low-frequency relief term normally uses (~0.15) these
    ///   rings come out at 36° and the pot renders as a machine screw — the corrugation tell
    ///   `stainless-steel` was eventually given `suppressMaps` for (DH-0002).
    /// - **Below**, the first cut derived this from the physical depth of a real throwing ring
    ///   (~0.3 mm on a 10 mm pitch ⇒ ~5°) and the rings were INVISIBLE in the render. The
    ///   arithmetic was right and the reasoning was wrong: this normal map is not competing with
    ///   nothing, it is competing with the material's OWN detail band, which was sitting at
    ///   ~10.6° and at several times the screen frequency, so it won twice. A geometrically
    ///   faithful term underneath a louder one is not a subtle version of the feature; it is the
    ///   feature not happening. The rings were raised to lead and the detail band cut to a
    ///   whisper (see `micro` in the generator) — the two are ONE balance, not two dials.
    ///
    /// Between those bounds it is a TASTE call, and it is Danny's: the first balanced cut put the
    /// satin default at ~4.4° and he read the rings as too pronounced (2026-09-09). This is that,
    /// softened ~40 %. Raise it for a more obviously hand-thrown pot, lower it toward a
    /// slip-cast one — but do not raise it without checking `micro` below, because the pair is
    /// what decides whether the rings read at all.
    public static let glazeRingAmplitude = 0.022
    /// Height→normal conversion strength for the glaze family. Paired with `glazeRingAmplitude`;
    /// changing one without the other changes the ring slope. Treat them as one decision.
    public static let glazeNormalStrength = 2.0

    /// **Glazed baked clay, in any colour, at any of the seven kiln finishes.**
    ///
    /// Four fields, layered the way a real pot is built:
    ///
    /// 1. **The throwing rings** — a spiral rib climbing the wall, at `glazeThrowingRingPitch`.
    ///    Drawn as a function of **v alone**, and that is the one directional feature that
    ///    survives this app's projection: a lathe's side wall has a mostly-radial normal, so
    ///    `HouseScene.vertex` box-projects it to `(z, y)` or `(x, y)` and **v is world height in
    ///    both**. A constant-v band is therefore a true horizontal ring all the way round the pot,
    ///    across the quadrant seam where the u axis switches from z to x. (On the top/foot caps
    ///    the dominant axis is Y and the same bands read as parallel lines — that is a couple of
    ///    square centimetres under the soil, and it is the price of not authoring cylindrical UVs
    ///    on a shared lathe primitive.)
    /// 2. **The glaze thickness** — a two-scale field, deeper in the ring grooves because that is
    ///    where a poured coat actually collects. Thick glaze reads deeper and glossier; thin glaze
    ///    washes out toward the clay.
    /// 3. **The body break** — wherever the coat runs thinnest, the CLAY shows, not a paler
    ///    version of the glaze. This is the whole reason a hand-glazed pot is not one flat colour,
    ///    and it is why the material takes a clay body constant at all.
    /// 4. **The recipe's own character** — iron speckle blooming through, or a crazed network,
    ///    per `GlazeFinish`.
    ///
    /// Everything a colour picker touches goes through `clampBand`, so no choice of glaze colour
    /// can leave the plausible dielectric albedo range. Roughness varies spatially in every
    /// finish (`TextureAudit.roughnessIsFlat` binds on this family), and the thickness field
    /// carries the macro tone spread the flat-colour tell requires.
    public static func glazedCeramic(size: Int = MaterialGenerator.bakeSize,
                                     params: GlazeParams = GlazeParams(),
                                     seed: UInt64 = 311) -> MaterialChannels {
        let f = params.finish
        var ch = MaterialChannels(size: size, category: .ceramic)
        let sh = seed

        // The clay body. On a GLAZED finish it is the potter's own stoneware and the picked
        // colour belongs to the coat on top. On `.unglazed` there is no coat, so the colour tints
        // the clay itself — desaturated toward earthenware, because a bisque pot fired in a
        // colouring slip is a blue-GREY clay, never a blue coat (see `GlazeParams`).
        let body = f.isGlazed ? glazeClayBody : clayTinted(by: params.color)

        // Rings per tile: an INTEGER, so `sin(2π·v·rings)` closes exactly at the v wrap and the
        // bake stays toroidal (the same constraint `marbleBeddingStretch` is an integer for).
        let rings = max(1, Int((glazeNominalRun / glazeThrowingRingPitch).rounded()))
        let ringAmp = glazeRingAmplitude * f.throwRelief

        // Voronoi distances come back in **cell-index units**, so every physical size above has
        // to be converted by its own lattice's cell count — a metre figure handed straight to
        // `smoothstep` is off by that count, which is how the crazing ended up 8 mm wide.
        let speckCells = max(2, Int((glazeNominalRun / glazeSpeckSpacing).rounded()))
        let speckRadius = (glazeSpeckSize / 2) / glazeNominalRun * Double(speckCells)
        let crazeCells = max(2, Int((glazeNominalRun / glazeCrazeCell).rounded()))
        // `f2 − f1` ≈ twice the distance to the boundary, so a half-width in distance is a
        // full width in this quantity.
        let crazeHalfWidth = (glazeCrazeWidth / 2) / glazeNominalRun * Double(crazeCells) * 2

        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)

                // 1 — the wheel. A slow tileable wander on the phase keeps it a hand's spiral
                // rather than a machined thread; the wander is a fraction of one ring, so the
                // rings never cross or double back.
                let wobble = Noise.fbmTiled(u, v, baseCells: 3, octaves: 2, seed: sh ^ 0x11) - 0.5
                let rib = sin(2 * Double.pi * (v * Double(rings) + 0.30 * wobble))   // −1 groove … +1 crest

                // 2 — how thick the coat pooled here. Zero spread on `.unglazed` (no coat), and
                // the groove term is why a ring reads as a darker line on a glazed pot and a
                // lighter one on a bare bisque body.
                let broad  = Noise.fbmTiled(u, v, baseCells: 3, octaves: 4, seed: sh) - 0.5
                let medium = Noise.fbmTiled(u, v, baseCells: 9, octaves: 3, seed: sh ^ 0x22) - 0.5
                let t = clamp01(0.62 + f.pooling * (broad * 1.15 + medium * 0.55 - rib * 0.18))

                // The clay body's own firing mottle — the uneven blush a kiln leaves on any
                // fired earth (the same term `terracotta` carries). It has to live on the BODY
                // rather than on the finished colour, because that is where it physically is:
                // under a thick coat it is invisible, under a thin one it shows through, and on
                // a bare bisque pot it is the only tonal structure there is.
                let fire = Noise.fbmTiled(u, v, baseCells: 4, octaves: 4, seed: sh ^ 0x9B)
                let bodyHere = body * (0.86 + 0.28 * fire)

                // 4 — the recipe's character. Both are the BODY asserting itself through the
                // coat, which is why they are sampled once and used in albedo and roughness alike.
                //
                // **A speck is `f1`, a crack is `f2 − f1`, and mixing them up draws the wrong
                // thing.** `voronoiTiled` returns the distances to the two nearest feature points.
                // `f2 − f1` is SMALL where the sample is equidistant from both — i.e. on a cell
                // BOUNDARY — so it draws a NETWORK. `f1` is small at a feature point itself, so
                // that is what draws a blob at a cell CENTRE.
                //
                // This started as a copy of `stoneware`'s `pow(max(0, 1 − (f2 − f1)·5), 8)`,
                // labelled there as "rare dark specks". It is not: it is a faint vein network,
                // and at stoneware's 0.10 amplitude nobody ever caught it. At the speckled
                // finish's 1.0 it drew a second crackle web over the first, and `reactive`
                // (speckle 0.35, crackle 0) rendered visibly CRAZED — a finish showing a feature
                // its recipe says it does not have (measured on the real frame, 2026-09-09).
                let spk = Noise.voronoiTiled(u, v, cells: speckCells, jitter: 0.9, seed: sh ^ 0x2C)
                let fleck = (1 - smoothstep(0, speckRadius, spk.f1)) * f.speckle
                let crz = Noise.voronoiTiled(u, v, cells: crazeCells, jitter: 0.85, seed: sh ^ 0x5F)
                let craze = (1 - smoothstep(0, crazeHalfWidth, crz.f2 - crz.f1)) * f.crackle

                // ── albedo ──
                // A thin coat is washed out — lighter AND less saturated, because there is less
                // pigment in the light path; a thick one is deep and saturated. One `t`, both
                // effects, so they cannot drift apart.
                let glaze = mix(desaturated(params.color, 0.35), params.color, t)
                          * (1.0 + (0.62 - t) * 0.34)
                // 3 — the break to the body. `.unglazed` short-circuits to bare clay rather
                // than falling out of the formula: at `bodyBreak == 1` the expression reduces to
                // `cover == t`, and `t` is a THICKNESS, which a pot with no coat on it does not
                // have — it sat at the field's 0.62 rest value and painted a bisque pot 62 %
                // glaze colour. Measured, 2026-09-09: an "unglazed" cobalt pot came out
                // (0.147, 0.171, 0.321), i.e. mostly cobalt.
                let cover = f.isGlazed ? clamp01(1 - f.bodyBreak * (1 - t)) : 0
                var c = mix(bodyHere, glaze, cover)
                c = mix(c, glazeIronSpeck, fleck * 0.85)
                c = mix(c, glazeCrazeStain, craze * 0.30)
                ch.albedo[ch.idx(x, y)] = clampBand(c)

                // ── roughness ──
                // Every term is the same physical story as its albedo term: thin coat scatters,
                // a filled groove is glassier, an iron speck is a matte inclusion, an open craze
                // line is a fissure, and the kiln leaves a fine orange peel on everything.
                let peel = Noise.fbmTiled(u, v, baseCells: 26, octaves: 3, seed: sh ^ 0x53) - 0.5
                var r = f.roughness
                if f.isGlazed {
                    r += (0.5 - t) * 0.16
                    r -= rib * 0.045 * f.throwRelief
                }
                r += fleck * 0.30 + craze * 0.14 + peel * 0.10
                ch.roughness[ch.idx(x, y)] = clamp01(r)

                // ── height ──
                // The rings are the only term with a slope worth having (see `glazeRingAmplitude`
                // for why the amplitude is derived rather than dialled); the rest is the gentle
                // swell of an uneven coat plus the two incised features.
                ch.height[ch.idx(x, y)] = clamp01(0.5 + rib * ringAmp
                                                  + (t - 0.62) * 0.10
                                                  - craze * 0.035 - fleck * 0.04)
            }
        }

        ch.clearcoat = f.clearcoat
        ch.clearcoatRoughness = f.clearcoatRoughness
        ch.deriveNormals(strength: glazeNormalStrength)
        // Fine kiln surface — the orange peel a glaze freezes with, or the grog tooth of a bare
        // bisque body. Stronger unglazed, for the same reason `PaintFinish.reliefScale` is:
        // a melt that flows buries what is under it.
        //
        // **A whisper, and the number is set against the RINGS rather than in isolation.** This
        // band first shipped at `0.28 + 0.55 · throwRelief` — ~0.53 at satin, above the shipped
        // `stoneware` (0.35) and `ceramic` (0.30) — and on a 50 mm close-up the pot came out
        // looking like painted stucco with no throwing rings on it at all. Measured: at that
        // strength the detail band's micro-tilt is ~10.6° against the rings' 4.5°, and it also
        // sits at several times their screen frequency. A glaze reads as a glaze when the rings
        // lead and this is the grain underneath them.
        let micro = 0.12 + 0.30 * f.throwRelief
        addMicroDetail(&ch, seed: sh ^ 0xD3, baseCells: 96, strength: micro,
                       occlusionStrength: micro * glazeDetailOcclusionShare)
        return ch
    }

    /// **The clay body stained by a colourant** — what `.unglazed` is made of.
    ///
    /// A colouring oxide or engobe worked into a clay body takes the colourant's HUE while
    /// keeping the clay's own VALUE: it is still fired earth, and there is no glassy coat to
    /// carry depth. So the tint is applied as a normalised RATIO (`colour ÷ its own luminance`,
    /// softened toward neutral), not as a blend toward the colour.
    ///
    /// Blending toward the colour is the obvious move and it is wrong, because a saturated glaze
    /// colour is much darker than a buff stoneware and the blend loses the hue before it loses
    /// the value: measured 2026-09-09, a 62 % blend of cobalt into this body came out
    /// (0.256, 0.238, 0.284) — a dead neutral grey with no blue left in it. The ratio form gives
    /// (0.43, 0.42, 0.52) at the same strength: a pale blue-grey clay, which is what a cobalt
    /// engobe on bisque actually looks like. It also does the right thing at both extremes for
    /// free — a near-white glaze colour leaves plain stoneware, and an oxblood one gives a red
    /// earthenware body.
    static func clayTinted(by color: Vec3, strength: Double = 0.35) -> Vec3 {
        let l = luma(color)
        guard l > 1e-4 else { return glazeClayBody }
        // Clamped so a very dark saturated colourant can't drive one channel off the scale
        // before `clampBand` has a chance to (and lose the hue doing it).
        let ratio = Vec3(clampRange(color.x / l, 0.45, 2.2),
                         clampRange(color.y / l, 0.45, 2.2),
                         clampRange(color.z / l, 0.45, 2.2))
        let tint = mix(Vec3(1, 1, 1), ratio, strength)
        return clampBand(Vec3(glazeClayBody.x * tint.x,
                              glazeClayBody.y * tint.y,
                              glazeClayBody.z * tint.z))
    }

    static func luma(_ c: Vec3) -> Double { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
    static func clampRange(_ v: Double, _ lo: Double, _ hi: Double) -> Double { Swift.max(lo, Swift.min(hi, v)) }

    /// Pull a colour `k` of the way toward its own luminance — the "less pigment in the light
    /// path" move a thin glaze and a coloured clay body both make. Rec. 709 weights.
    static func desaturated(_ c: Vec3, _ k: Double) -> Vec3 {
        let l = luma(c)
        return mix(c, Vec3(l, l, l), k)
    }
}
