import simd
import Foundation

extension MaterialGenerator {
    /// Customizable terrazzo parameters — the MATERIALS_AND_TEXTURES §7 "customization
    /// spine" (seed + colors + scale + finish), carried forward from Roominate's
    /// `customizationDescriptors`. Defaults reproduce the muted architectural look.
    public struct TerrazzoParams: Equatable, Hashable, Sendable, Codable {
        /// The aggregate chip palette, **ordered by prominence**: a chip picks its colour with a
        /// linearly-decreasing weight over this array (entry 0 is roughly 4× as common as the last
        /// of eight), plus a ±6% per-chip value variation. Real terrazzo is a mostly-neutral field
        /// with saturated accents scattered through it, and a uniform pick over an 8-colour palette
        /// makes confetti instead. Ordering IS the weighting, which also matches the customizer:
        /// it exposes the first three as "Chip 1…3" precisely because they are the dominant ones.
        public var chipColors: [Vec3]
        /// The cement binder color between the chips.
        public var matrixColor: Vec3
        /// Overall chip-size multiplier (0.5 = fine, 2 = chunky).
        public var chipScale: Double
        /// **How blunt a chip's corners are** — the fillet radius as a fraction of the chip's
        /// circumradius. 0 = smashed glass (sharp points), 0.15 = crushed-and-ground aggregate
        /// (the default: visible flats, knocked-off corners), 0.5 = tumbled river pebble. This is
        /// the shape knob: past ~0.4 the flats disappear and the chips read as ovals again.
        public var chipRounding: Double
        /// 0 = matte/honed … 1 = high-gloss polish (drives roughness + the clearcoat lobe).
        public var polish: Double
        /// Redraws the whole scatter.
        public var seed: UInt64

        public init(chipColors: [Vec3] = TerrazzoParams.classicPalette,
                    matrixColor: Vec3 = TerrazzoParams.classicMatrix,
                    chipScale: Double = 1, chipRounding: Double = 0.15,
                    polish: Double = 0.7, seed: UInt64 = 1) {
            self.chipColors = chipColors.isEmpty ? TerrazzoParams.classicPalette : chipColors
            self.matrixColor = matrixColor; self.chipScale = chipScale
            self.chipRounding = chipRounding
            self.polish = polish; self.seed = seed
        }

        /// Tolerant decode — a document saved before a knob existed still opens, wearing that
        /// knob's default rather than failing the whole load (same contract as `DocumentMaterial`).
        public init(from decoder: Decoder) throws {
            let d = try decoder.container(keyedBy: CodingKeys.self)
            let def = TerrazzoParams()
            let cols = try d.decodeIfPresent([Vec3].self, forKey: .chipColors) ?? def.chipColors
            self.init(chipColors: cols,
                      matrixColor: try d.decodeIfPresent(Vec3.self, forKey: .matrixColor) ?? def.matrixColor,
                      chipScale: try d.decodeIfPresent(Double.self, forKey: .chipScale) ?? def.chipScale,
                      chipRounding: try d.decodeIfPresent(Double.self, forKey: .chipRounding) ?? def.chipRounding,
                      polish: try d.decodeIfPresent(Double.self, forKey: .polish) ?? def.polish,
                      seed: try d.decodeIfPresent(UInt64.self, forKey: .seed) ?? def.seed)
        }

        /// The default palette, prominence-ordered (see `chipColors`). Values are LINEAR albedo —
        /// the comment beside each is the sRGB it displays as. All inside the dielectric band
        /// (`clampBand`, 0.045…0.93), which is why the "black" fleck is a very dark charcoal:
        /// a true black is not a physically plausible dielectric albedo.
        public static let classicPalette: [Vec3] = [
            Vec3(0.75, 0.73, 0.68),   // warm off-white marble   #ddd9d3
            Vec3(0.49, 0.58, 0.71),   // pale slate blue         #b8c6d8
            Vec3(0.69, 0.48, 0.26),   // warm sand               #d8b78c
            Vec3(0.05, 0.05, 0.06),   // charcoal (the pepper)   #292a2c
            Vec3(0.41, 0.37, 0.24),   // olive khaki             #aaa286
            Vec3(0.81, 0.55, 0.48),   // dusty pink              #e8c2b8
            Vec3(0.34, 0.05, 0.05),   // brick red               #9c2a2a
            Vec3(0.85, 0.48, 0.06),   // amber                   #edb642
        ]

        /// The cement binder — a warm near-white (sRGB ≈ #f2efe9). The old 0.66 grey binder is
        /// half the reason the material read as "confetti on concrete": a poured terrazzo matrix
        /// is a pale cement, and the chips read as chips because they sit DARKER than it.
        public static let classicMatrix = Vec3(0.76, 0.74, 0.70)

        /// Curated starting points (Roominate-style named looks) — pick one, then tweak.
        public static let presets: [(name: String, params: TerrazzoParams)] = [
            ("Classic", .init()),
            ("Venetian", .init(chipColors: [Vec3(0.90, 0.88, 0.83), Vec3(0.78, 0.66, 0.42),
                                            Vec3(0.66, 0.40, 0.30), Vec3(0.55, 0.46, 0.38)],
                                matrixColor: Vec3(0.80, 0.76, 0.69), polish: 0.85)),
            ("Monochrome", .init(chipColors: [Vec3(0.90, 0.90, 0.90), Vec3(0.22, 0.22, 0.24),
                                              Vec3(0.55, 0.55, 0.57), Vec3(0.72, 0.72, 0.73)],
                                  matrixColor: Vec3(0.70, 0.70, 0.71), polish: 0.7)),
            ("Coastal", .init(chipColors: [Vec3(0.88, 0.90, 0.90), Vec3(0.36, 0.50, 0.58),
                                           Vec3(0.50, 0.62, 0.56), Vec3(0.74, 0.78, 0.74)],
                               matrixColor: Vec3(0.82, 0.82, 0.80), polish: 0.75)),
            ("Bold", .init(chipColors: [Vec3(0.86, 0.20, 0.24), Vec3(0.18, 0.42, 0.66),
                                        Vec3(0.92, 0.78, 0.24), Vec3(0.90, 0.90, 0.88)],
                            matrixColor: Vec3(0.16, 0.17, 0.20), polish: 0.85)),
        ]
    }

    /// Back-compat / convenience: terrazzo by seed (no `params` overload to disambiguate).
    public static func terrazzo(size: Int = MaterialGenerator.bakeSize, seed: UInt64) -> MaterialChannels {
        terrazzo(size: size, params: TerrazzoParams(seed: seed))
    }

    /// One **aggregate grade**: a crushed-chip size band in real MILLIMETRES, plus the fraction of
    /// the floor that grade covers. This is how terrazzo is actually specified (aggregate is sold
    /// by graded size), and it is the single source for both chip size and chip COUNT — the count
    /// is derived (`coverage × area ÷ mean chip area`), never typed. Two consequences fall out for
    /// free: the "Chip size" slider changes size without thinning the field, and a bake at a
    /// different world span gets the right number of chips instead of the same number, bigger.
    public struct AggregateGrade: Equatable, Hashable, Sendable {
        /// Chip circumradius band, millimetres.
        public var minRadiusMM: Double
        public var maxRadiusMM: Double
        /// Fraction of the surface this grade covers.
        public var coverage: Double
        /// How strongly this grade skews to the palette's darkest colour (0 = palette weighting,
        /// 1 = always). The finest aggregate in real terrazzo is crusher dust and reads as dark
        /// pepper, which is a large part of what makes the field look like stone rather than dots.
        public var darkBias: Double

        public init(_ minRadiusMM: Double, _ maxRadiusMM: Double,
                    coverage: Double, darkBias: Double = 0) {
            self.minRadiusMM = minRadiusMM; self.maxRadiusMM = maxRadiusMM
            self.coverage = coverage; self.darkBias = darkBias
        }
    }

    /// The default aggregate mix — a Venetian-ish grading: a few statement chips, a working
    /// population of mid chips, and a pepper of fines. Total coverage 0.34, i.e. two thirds of the
    /// surface is cement, which is what a poured floor looks like.
    public static let terrazzoGrades: [AggregateGrade] = [
        .init(12, 22, coverage: 0.15),                     // statement chips, ~2.5–4.5 cm across
        .init(6,  12, coverage: 0.14),                     // the working mid grade
        .init(3,   6, coverage: 0.08, darkBias: 0.25),     // fines
        .init(1,   3, coverage: 0.02, darkBias: 0.55),     // crusher dust — the dark pepper
    ]

    /// **World metres one terrazzo bake spans, when the surface lets the material choose it.**
    ///
    /// The same lever as `TileLayout.targetRunMeters`, and here it is not a taste call but a
    /// resolution one: a 512 bake stretched over a room floor's nominal 2 m run gives 3.9 mm per
    /// texel, so a real 12 mm chip is THREE TEXELS and the whole fine grade is sub-texel. That is
    /// why the chips had to be 10–27 cm boulders to be drawable at all. At 0.6 m the texel is
    /// 1.2 mm, which draws the full grading above with the finest chips still ~2 texels wide.
    ///
    /// A short period is normally the thing you avoid on a stochastic material — but terrazzo is
    /// stochastic precisely in the sense the hex de-repeat blend handles best (no coherent pattern
    /// to double-print), and a uniform random speckle has no landmark to clone-spot.
    public static let terrazzoRunMeters: Double = 0.6

    /// The run a terrazzo bake gets on a given surface: our own period where the mesh takes its
    /// UVs from the material, and the mesh's period where it baked them earlier and would render
    /// a size we never chose.
    public static func terrazzoRun(on tiling: SurfaceTiling) -> Double {
        tiling.isAdjustable ? Swift.min(tiling.metresPerRepeat, terrazzoRunMeters)
                            : tiling.metresPerRepeat
    }

    /// Terrazzo: crushed stone chips scattered in a cement matrix — the §1 hero. The chip is an
    /// **irregular straight-edged fragment**, not an ellipse: a 7-sector star polygon whose sector
    /// radii are jittered per chip, so every chip has the corners, flats and notches of crushed
    /// rock. (It was an oriented ellipse, and a field of soft ovals is what made Danny's "the
    /// chips are ovals… it is NUTS" — a smooth closed curve reads as a *bean*, and no amount of
    /// size or colour tuning fixes a wrong silhouette.)
    ///
    /// Size and count come from `terrazzoGrades` in MILLIMETRES against `runMeters`, so the chips
    /// are a real physical size on the floor and the count follows the coverage. Chips are scattered
    /// with a Poisson-style separation test — real chips displace each other in the pour, and
    /// unconstrained scatter piles them into blobs that read as one big chip.
    ///
    /// Seamless (toroidal chip distance and separation); a spatial-hash grid keeps the bake fast.
    /// Fully procedural and `TerrazzoParams`-customizable (colors/scale/finish).
    public static func terrazzo(size: Int = MaterialGenerator.bakeSize,
                                params: TerrazzoParams = .init(),
                                runMeters: Double = MaterialGenerator.terrazzoRunMeters) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .terrazzo)
        let palette = params.chipColors
        let darkest = palette.min { TextureAudit.luma($0) < TextureAudit.luma($1) } ?? palette[0]

        // Seeded RNG over the hash mixer.
        var s = params.seed &+ 0x9E37_79B9_7F4A_7C15
        func rnd() -> Double { s = Noise.mix(s); return Noise.unit(s) }

        // A scattered chip: centre, circumradius, cross-axis squash, orientation, colour. The
        // outline is stored as HALF-PLANES in `edges` — one outward normal + offset per side, in
        // unit-circumradius chip space, packed one chip after another and addressed by
        // `base`/`sides`. A pixel's inside-distance is then the (smoothed) minimum over the sides,
        // which is what rounds every corner off; see `cornerRound`.
        struct Chip { var x, y, r, aspect, ca, sa: Double; var color: Vec3; var base, sides: Int }
        var chips: [Chip] = []
        var edges: [(nx: Double, ny: Double, d: Double)] = []

        // The grid does double duty: the separation test during scatter, and the per-pixel chip
        // lookup below. A chip is registered in every cell its bounding box touches (toroidally),
        // which is exactly the condition two overlapping chips must share, so the separation test
        // can never miss a neighbour.
        let runMM = Swift.max(runMeters, 1e-3) * 1000
        let coarsestFrac = terrazzoGrades.map(\.maxRadiusMM).max()! * Swift.max(params.chipScale, 0.1) / runMM
        let gridN = Swift.max(8, Swift.min(64, Int(0.5 / Swift.max(coarsestFrac, 1e-3))))
        var grid = [[Int32]](repeating: [], count: gridN * gridN)
        func cell(_ gx: Int, _ gy: Int) -> Int { Noise.wrap(gy, gridN) * gridN + Noise.wrap(gx, gridN) }
        func cells(x: Double, y: Double, r: Double, _ body: (Int) -> Void) {
            let g = Double(gridN)
            for gx in Int(floor((x - r) * g))...Int(floor((x + r) * g)) {
                for gy in Int(floor((y - r) * g))...Int(floor((y + r) * g)) { body(cell(gx, gy)) }
            }
        }
        func toroidal(_ d: Double) -> Double { d > 0.5 ? d - 1 : (d < -0.5 ? d + 1 : d) }

        /// Corner fillet radius, as a fraction of a chip's circumradius (`chipRounding`). Clamped
        /// off zero because it is the smooth-min's blend width and divides.
        let cornerRound = Swift.max(params.chipRounding, 1e-4)

        /// **Crushed rock is convex, and its corners are knocked off.** Two shape failures, one
        /// after the other, both reported as "ninja star … jagged shapes" (Danny):
        ///  1. Corners jittered freely produce REFLEX vertices, and a reflex vertex is a spike. So
        ///     the outline is the CONVEX HULL of the jittered corners — a repeated sweep dropping
        ///     every non-left turn (a Graham scan, with the sort already done for it).
        ///  2. A convex polygon is still *pointy*, and terrazzo aggregate is tumbled and then ground
        ///     flat: the chips in a real floor have rounded corners and only slightly curved edges.
        ///     So the shape is not filled as a polygon at all — the hull becomes a set of HALF-PLANES
        ///     and the fill takes a smooth-min over them (`cornerRound`), which is a polygon with
        ///     every corner filleted, by construction and at every size.
        ///
        /// Returns outward-normal half-planes in unit-circumradius space; inside is `d - n·p ≥ 0`.
        func convexEdges(_ pts: [(a: Double, r: Double)]) -> [(nx: Double, ny: Double, d: Double)] {
            var hull = pts.map { (x: $0.r * cos($0.a), y: $0.r * sin($0.a)) }
            var dropped = true
            while hull.count > 3 && dropped {
                dropped = false
                var i = 0
                while i < hull.count && hull.count > 3 {
                    let p = hull[(i + hull.count - 1) % hull.count]
                    let q = hull[i], r = hull[(i + 1) % hull.count]
                    if (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x) <= 1e-12 {
                        hull.remove(at: i); dropped = true
                    } else { i += 1 }
                }
            }
            return (0..<hull.count).map { i in
                let p = hull[i], q = hull[(i + 1) % hull.count]
                let ex = q.x - p.x, ey = q.y - p.y
                let len = Swift.max((ex * ex + ey * ey).squareRoot(), 1e-9)
                let nx = ey / len, ny = -ex / len          // outward normal of a CCW hull
                // Pushed out by the fillet radius, so rounding the corners does not also shrink
                // the chip — the smooth-min erodes by up to `cornerRound` and this gives it back.
                return (nx: nx, ny: ny, d: nx * p.x + ny * p.y + cornerRound * 0.5)
            }
        }

        /// Centres must be at least this far apart, as a fraction of the two circumradii summed.
        /// **At 1.0 the circumcircles are disjoint, so no two chips can overlap at all** — and that
        /// is not a density preference, it is what keeps every chip convex ON SCREEN. Two chips
        /// that overlap fuse into one silhouette with a notch where the smaller one bites the
        /// larger, which is the "ninja star" again by another route: hulling the outline is
        /// pointless if the thing you SEE is a union of two of them.
        let separation = 1.0
        /// A chip smaller than about a texel cannot be drawn; it aliases into pepper noise instead.
        /// Flooring the radius here is also what keeps the chip COUNT finite as the run grows: the
        /// count is coverage ÷ chip area, so a grade that would need 14 000 sub-texel specks at a
        /// 2 m run instead draws 2 500 texel-sized ones.
        let drawableMM = 1.2 * runMM / Double(size)

        for grade in terrazzoGrades {
            let lo = Swift.max(grade.minRadiusMM * params.chipScale, drawableMM)
            let hi = Swift.max(grade.maxRadiusMM * params.chipScale, lo)
            // Mean drawn area of a jittered 7-gon of circumradius r, measured against π r²: the
            // sector radii average 0.8 and the straight chords cut the corners → ≈0.58.
            let meanArea = Double.pi * pow((lo + hi) * 0.5, 2) * 0.65
            let count = Int((grade.coverage * runMM * runMM / meanArea).rounded())
            for _ in 0..<count {
                var col = palette[Int(Swift.min(rnd(), rnd()) * Double(palette.count)) % palette.count]
                if rnd() < grade.darkBias { col = darkest }
                col = col * (0.94 + 0.12 * rnd())                    // ±6% per-chip value variation
                let r = (lo + rnd() * (hi - lo)) / runMM             // millimetres → tile fraction
                let aspect = 0.58 + 0.42 * rnd()                     // some shards, some equant
                let ang = rnd() * .pi
                let sides = 5 + Int(rnd() * 5.0)                     // 5…9
                // Up to 20 sites before this chip is dropped, and a chip that cannot be seated is one the
                // pour had no room for either — the achieved coverage tops out where the packing does.
                var placed = false
                for _ in 0..<20 where !placed {
                    let x = rnd(), y = rnd()
                    var clear = true
                    cells(x: x, y: y, r: r) { c in
                        for oi in grid[c] where clear {
                            let o = chips[Int(oi)]
                            let dx = toroidal(x - o.x), dy = toroidal(y - o.y)
                            let gap = (r + o.r) * separation
                            if dx * dx + dy * dy < gap * gap { clear = false }
                        }
                    }
                    guard clear else { continue }
                    placed = true
                    // Corner k sits in its own angular slice, jittered up to ±35% of the slice, so
                    // the corners stay angle-sorted while the edges between them come out unequal —
                    // evenly-spaced corners read as a regular polygon stamped over and over.
                    let slice = 2 * Double.pi / Double(sides)
                    let outline = convexEdges((0..<sides).map { k in
                        (a: (Double(k) + 0.35 * (rnd() * 2 - 1)) * slice, r: 0.72 + 0.28 * rnd())
                    })
                    let idx = Int32(chips.count)
                    chips.append(Chip(x: x, y: y, r: r, aspect: aspect,
                                      ca: cos(ang), sa: sin(ang), color: clampBand(col),
                                      base: edges.count, sides: outline.count))
                    edges.append(contentsOf: outline)
                    cells(x: x, y: y, r: r) { grid[$0].append(idx) }
                }
            }
        }

        let chipRough = mix(0.30, 0.06, params.polish)               // polished aggregate
        let binderRough = mix(0.42, 0.16, params.polish)             // matte cement
        /// Edge softening, in tile fractions: one texel, so a chip edge is crisp but not stair-stepped.
        let aa = 1.0 / Double(size)

        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size), v = (Double(y) + 0.5) / Double(size)
                // Topmost containing chip (fine grades are scattered last → highest index wins).
                var hitColor = params.matrixColor
                var chipMask = 0.0          // 1 inside a chip, fading to 0 across its rim
                var bestIdx = -1
                for ci32 in grid[cell(Int(u * Double(gridN)), Int(v * Double(gridN)))] {
                    let ci = Int(ci32)
                    guard ci > bestIdx else { continue }
                    let c = chips[ci]
                    let dx = toroidal(u - c.x), dy = toroidal(v - c.y)
                    guard abs(dx) <= c.r + aa, abs(dy) <= c.r + aa else { continue }  // cheap bbox reject
                    let lx = (dx * c.ca + dy * c.sa) / c.r                        // chip-local…
                    let ly = (-dx * c.sa + dy * c.ca) / (c.aspect * c.r)          // …de-squashed, unit
                    let lim = 1 + aa / c.r                                        // one texel of slack
                    guard lx * lx + ly * ly <= lim * lim else { continue }
                    // Inside-distance = the SMOOTH minimum over the chip's half-planes. A hard min
                    // would be the polygon, with its corners; blending the two nearest planes over
                    // `cornerRound` fillets every corner instead — a rounded convex fragment, which
                    // is what a tumbled-then-ground aggregate chip is.
                    var s = Double.greatestFiniteMagnitude
                    for e in edges[c.base..<(c.base + c.sides)] {
                        let di = e.d - (e.nx * lx + e.ny * ly)
                        let h = Swift.max(cornerRound - abs(s - di), 0) / cornerRound
                        s = Swift.min(s, di) - h * h * cornerRound * 0.25
                    }
                    let m = clamp01(s * c.r / aa + 0.5)
                    if m > 0 { bestIdx = ci; hitColor = c.color; chipMask = m }
                }
                // Macro tonal drift so the binder + same-color chips aren't dead flat (§4/§5).
                let macro = 0.93 + 0.14 * Noise.fbmTiled(u, v, baseCells: 3, octaves: 4, seed: params.seed ^ 0xA5)
                let albedo = mix(params.matrixColor, hitColor, chipMask) * macro
                ch.albedo[ch.idx(x, y)] = clampBand(albedo)

                let pore = (Noise.fbmTiled(u, v, baseCells: 24, octaves: 3, seed: params.seed ^ 0x5C) - 0.5) * 0.03
                ch.roughness[ch.idx(x, y)] = clamp01(mix(binderRough, chipRough, chipMask) + pore)

                // A polished floor is GROUND FLAT — the chips are cut through, flush with the
                // binder, and only an unpolished/honed floor leaves them standing proud. With
                // chips now at their real size that plateau is everywhere, so the old half-height
                // step would have embossed the whole surface into gravel.
                let relief = Noise.fbmTiled(u, v, baseCells: 32, octaves: 3, seed: params.seed ^ 0x3F) * 0.04
                ch.height[ch.idx(x, y)] = clamp01(0.5 + chipMask * 0.10 * (1 - params.polish) + relief)
            }
        }
        ch.clearcoat = 0.15 + 0.35 * params.polish                   // the §1 wet polish sheen
        ch.deriveNormals(strength: 6)
        // Faint sub-chip micro-relief so the polished aggregate isn't a dead-flat plastic
        // sheet at grazing range. Kept subtle — terrazzo IS polished, not a rough stone.
        addMicroDetail(&ch, seed: params.seed ^ 0xB1, baseCells: 130, strength: 0.30,
                       occlusionStrength: 0.30 * glazeDetailOcclusionShare)
        return ch
    }
}
