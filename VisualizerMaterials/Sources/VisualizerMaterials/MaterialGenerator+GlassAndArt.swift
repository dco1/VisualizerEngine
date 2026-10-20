import simd
import Foundation

extension MaterialGenerator {
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
