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
        public static let classicMatrix = Vec3(0.86, 0.84, 0.80)

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

    /// **The paint tile is SMALL on purpose, and the size is load-bearing.** Paint has no macro
    /// structure — no plank, no grout, no weave — so every texel a big tile spends is spent on
    /// nothing, and the thing paint *does* have (roller stipple, 0.5–3 mm) falls below what the
    /// tile can represent. Measured 2026-08-15: at the old 3.0 m repeat and 512², one texel was
    /// **5.9 mm**, so orange peel was not merely faint, it was UNREPRESENTABLE — Nyquist needs two
    /// texels per feature — and every noise band below sat at decimetre scale (tooth 10.7 cm,
    /// roller micro 18.75 cm, macro drift 1 m).
    ///
    /// That is the root cause of the two "it looks like a sponge" reports. Decimetre-scale
    /// variation reads as BLOTCH at any amplitude, so both times the fix was to cut amplitude —
    /// albedo variance to ~1 %, relief to 0.3, micro-occlusion to zero — until the blotch was gone
    /// and so was every trace of surface. Frequency was the defect; amplitude was the symptom.
    ///
    /// At the 0.6 m repeat `MaterialResolver.paintTiling` now uses, one texel is 1.17 mm and the
    /// same cell counts land where paint actually lives:
    ///
    /// | band | cells | feature size | carries |
    /// |---|---|---|---|
    /// | drift | 3 | 20 cm | macro tone (whisper — the sponge lived here) |
    /// | tooth | 28 | 2.1 cm | albedo + roughness breakup |
    /// | micro | 16 | 3.75 cm | roller lap → macro normal |
    /// | pore | 64, ×8 UV | 1.2 mm | **orange peel** → detail normal + micro-occlusion |
    ///
    /// Changing `paintTiling` re-scales all four. Treat the two as one decision.
    ///
    /// Painted plaster wall — the simplest member of the library, and the workhorse
    /// surface of any interior. Near-flat warm color with a low-frequency macro tone
    /// and a faint trowel normal, plus the micro roughness variation real paint always
    /// carries (so it never trips the flat-roughness tell #3).
    /// `finish` is the single source for roughness, clearcoat and relief amplitude — the three
    /// move together because they are all consequences of how much the film levels (see
    /// `PaintFinish`). `reliefStrength` overrides only the relief, for a SUBSTRATE that is
    /// smoother than drywall at the same sheen: painted MDF cabinetry is the case that needs it,
    /// because at close range a wall's full tooth reads as orange-peel blotch on a cabinet face.
    public static func paint(size: Int = MaterialGenerator.bakeSize, color: Vec3 = Vec3(0.82, 0.80, 0.76),
                             finish: PaintFinish = .default, reliefStrength: Double? = nil,
                             seed: UInt64 = 2) -> MaterialChannels {
        let finishRoughness = finish.roughness
        let relief = reliefStrength ?? finish.reliefScale
        var ch = MaterialChannels(size: size, category: .paint)
        var detailH = [Double](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Paint albedo is NEARLY UNIFORM IN TONE, and that is the whole point.
                //
                // Twice now a tonal-variation term here has been the thing Danny saw: the
                // 2026-07-13 "grimy stucco" (a 2-cell drift putting all the variance at
                // metre scale) and the 2026-08-05 "painted with a sponge… parts look like
                // black mold" (the 0.150 `tooth` that replaced it at ~9 cm scale). The first
                // fix only MOVED the variance — "same total albedo SD, gate stays green" —
                // because `TextureAudit.macroCVFloor` failed any material whose albedo SD ÷
                // mean was under 1.2 %, so the generator was obliged to put blotch SOMEWHERE.
                //
                // Localized on the real Metal path (`HouseRenderBridgeGPUTests_WallSplotch`):
                // zeroing `tooth` collapsed the bake's albedo variation 6.7× and visibly
                // de-sponged the wall, while shadows, SSAO, the hex anti-tiling blend and the
                // relief normals were each ablated and cleared. So the gate was relaxed for
                // paint alone (`TextureAudit.flatColorIsCorrect`) and the variance MOVED OUT
                // OF TONE ENTIRELY into roughness, where real rolled paint actually carries
                // it: a wall reads as paint because it scatters unevenly, not because it is
                // blotchy. Both terms here are now a whisper — enough to break a mathematically
                // dead flat fill, far below the threshold where the eye reads "sponge".
                //
                // The roughness tell (`roughnessIsFlat`) still binds on paint deliberately, so
                // this cannot decay into a flat plastic scalar.
                let drift = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: seed) - 0.5
                let tooth = Noise.fbmTiled(u, v, baseCells: 28, octaves: 3, seed: seed ^ 0x9E) - 0.5
                let macro = 1.0 + 0.006 * drift + 0.012 * tooth
                ch.albedo[ch.idx(x, y)] = clampBand(color * macro)
                // Roller micro relief: the roughness breakup + a subtle normal. This is where
                // paint's life lives now — see the tone note above the albedo terms.
                let micro = Noise.fbmTiled(u, v, baseCells: 16, octaves: 3, seed: seed ^ 0x7B)
                ch.height[ch.idx(x, y)] = micro
                // fine-grain brush / roller stipple — the detail normal AND the mm-scale half of
                // the roughness breakup.
                let pore = Noise.fbmTiled(u, v, baseCells: 64, octaves: 2, seed: seed ^ 0xC3)
                detailH[ch.idx(x, y)] = pore
                // Roughness carries the finish's variation at BOTH bands. The cm terms (micro
                // 3.75 cm, tooth 2.1 cm) were the only ones here, and they are the scale you read
                // across a room; `pore` at 1.2 mm is the scale you read from a metre away, which
                // is where a wall either looks painted or looks printed. A wall reads as paint
                // because it scatters unevenly under a grazing light — that is a roughness story
                // first and a normal story second, which is why this band goes here and not into
                // more relief.
                ch.roughness[ch.idx(x, y)] = clamp01(finishRoughness + (micro - 0.5) * 0.16
                                                     + tooth * 0.10
                                                     + (pore - 0.5) * 0.10 * relief)
            }
        }
        // The polish lobe over the pigment — 0 for flat/matte, real for the sheens.
        ch.clearcoat = finish.clearcoat
        // Roller lap at 3.75 cm (see the tile table above). Scaled by the finish, because a
        // gloss enamel flows level and a flat paint keeps the stipple it was rolled with.
        ch.deriveNormals(strength: 0.8 * relief)
        // Detail normal from the pore/stipple band, sampled at ×8 UV ⇒ **1.2 mm** on the wall:
        // this is the orange peel, and at the old 3.0 m tile it was 5.9 mm and could not be one.
        //
        // Paint's micro-occlusion is back ON, and the measurement that switched it off is not
        // being ignored — it is being re-run at a frequency that did not exist when it was taken.
        // That reading (blotch SD 0.07 → 0.25) was made with the detail band at ~4.7 cm, where
        // occlusion shades centimetre patches and the eye correctly calls it mould. At 1.2 mm the
        // same term is the diffuse read of surface tooth, which is the ONLY thing that makes
        // paint look like paint rather than a coloured card — the detail NORMAL alone cannot
        // supply it, having measured 0.98–1.03 on paint (invisible: it is a specular-band effect
        // and paint's specular is weak).
        //
        // Amount is the LIBRARY DEFAULT — `setDetailRelief` uses the normal's own strength when
        // no occlusion strength is given, and paint no longer asks for less. It first shipped at
        // half that (`0.35 * relief`, a 2.1 % mean darkening) out of caution about the mould
        // report, and the `paintTooth` station showed exactly what caution bought: at one metre
        // the panel was still a flat card beside its plaster neighbour. Half of a term that was
        // already subtle is not a conservative version of the fix, it is the fix not happening.
        // Scaled by the finish throughout, so a gloss door still gets almost none.
        //
        // The wall-splotch gate (`HouseRenderBridgeGPUTests_WallSplotch`) is the tripwire: if
        // this reintroduces tone structure at a scale the eye reads as sponge, that number moves.
        setDetailRelief(&ch, height: detailH, strength: 0.7 * relief)
        return ch
    }

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
        var ch = MaterialChannels(size: size, category: .tile)
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
    public static let ceramicAlbedo    = Vec3(0.905, 0.910, 0.915)
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
        var ch = MaterialChannels(size: size, category: .tile)
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
    public nonisolated(unsafe) static var brushStreakRoughnessAmp: Double = 0.22

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
    public nonisolated(unsafe) static var brushStreakHeightAmp: Double = 0.20

    public static func brushedSteel(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 6,
                                    base baseIn: Vec3 = Vec3(0.58, 0.585, 0.59)) -> MaterialChannels {
        let legacy = legacyTonalBrushStreakForTest
        let base = legacy ? Vec3(0.66, 0.64, 0.60) : baseIn
        var ch = MaterialChannels(size: size, category: .metal)
        var grain = [Vec2](repeating: Vec2(1, 0), count: size * size)   // brush runs along X
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // streaks: high frequency ACROSS the brush (v), low along it (u)
                let streak = Noise.fbmTiled(u * 2, v * 24, baseCells: 4, octaves: 3, seed: seed)
                let fine = Noise.fbmTiled(u * 1, v * 64, baseCells: 6, octaves: 2, seed: seed ^ 0x55)

                // FLAT albedo — a brushed metal's grain is a roughness/normal phenomenon, and
                // its albedo is its F0. See the note above for the measurement that says so.
                ch.albedo[ch.idx(x, y)] = clampBand(legacy ? base * (0.88 + 0.20 * streak) : base)
                // …so roughness carries the whole brush. Amplitude raised 0.22 → 0.30 across
                // the grain to take over what the tone was doing; the fine cross-hatch and the
                // 0.28 mean are unchanged, so the satin level is where it was.
                ch.roughness[ch.idx(x, y)] = clamp01(
                    0.28 + (streak - 0.5) * (legacy ? 0.22 : brushStreakRoughnessAmp)
                         + (fine - 0.5) * 0.10)
                ch.height[ch.idx(x, y)] = clamp01(
                    0.5 + (streak - 0.5) * (legacy ? 0.5 : brushStreakHeightAmp))
                grain[ch.idx(x, y)] = Vec2(1, 0)
            }
        }
        ch.grainTangent = grain
        ch.deriveNormals(strength: 2)
        // Fine satin micro-tooth between the brush grooves — a real brushed-steel face
        // isn't a mirror at close range. Subtle so it doesn't drown the anisotropic streak.
        addMicroDetail(&ch, seed: seed ^ 0xB2, baseCells: 100, strength: 0.30)
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
    /// v-directional (`v * 24`, integer for a seamless wrap), with darker rain-runs and the odd
    /// brighter fresh-oxide bloom. Isotropic (no grain tangent): the streaks are tone, not a
    /// brushed grain the engine should stretch a highlight along.
    public static func corten(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 211,
                              base: Vec3 = Vec3(0.44, 0.23, 0.13),      // warm rust orange-brown
                              bloom: Vec3 = Vec3(0.58, 0.36, 0.21),     // lighter fresh-oxide flush
                              runoff: Vec3 = Vec3(0.24, 0.13, 0.09)     // dark rain-streak stain
    ) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Broad blotchy patina + vertical rain-runs + fine pit mottle.
                let patch  = Noise.fbmTiled(u, v, baseCells: 3, octaves: 4, seed: seed)
                let streak = Noise.fbmTiled(u, v * 24.0, baseCells: 4, octaves: 3, seed: seed ^ 0x5C)
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
        ch.deriveNormals(strength: 2.5)
        addMicroDetail(&ch, seed: seed ^ 0xE7, baseCells: 88, octaves: 2, strength: 0.80)   // clay grain tooth
        return ch
    }

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
                let warp = Noise.fbmTiled(u * 48, v, baseCells: 4, octaves: 2, seed: seed)
                let weft = Noise.fbmTiled(u, v * 48, baseCells: 4, octaves: 2, seed: seed ^ 0x5E)
                let nap = (warp + weft) * 0.5
                let macro = 0.82 + 0.36 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed ^ 0x6F)
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * (0.85 + 0.3 * nap))
                ch.roughness[ch.idx(x, y)] = clamp01(0.84 + (nap - 0.5) * 0.16)   // matte cloth, nap-broken
                ch.height[ch.idx(x, y)] = nap
            }
        }
        ch.sheen = 0.85                            // velvet is sheen-dominated
        ch.deriveNormals(strength: 2)
        // Very fine sub-thread nap so the pile catches grazing light as fuzz, not plastic.
        addMicroDetail(&ch, seed: seed ^ 0xB3, baseCells: 120, strength: 0.40)
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
        ch.deriveNormals(strength: 4)
        // Soft fibrous fuzz over the chunky loops — the between-loop wool haze at close range.
        addMicroDetail(&ch, seed: seed ^ 0xB4, baseCells: 60, strength: 0.70)
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
                grain *= 0.90 + 0.18 * Noise.unit(Noise.mix(cell.cellId))                // per-grain value
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

    /// Venetian plaster — the classic troweled-and-burnished wall finish: fine
    /// cloud-like streaking laid down in opposing trowel strokes, then polished
    /// to a low semi-gloss. Albedo: warm off-white with very subtle warm streaks
    /// (looks different from plain paint because of the directional micro-sheen).
    /// Roughness: 0.18–0.32 — much smoother than matte paint, slightly patchy.
    /// Clearcoat 0.25: the burnished polish layer catches light at grazing angles.
    public static func venetianPlaster(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 51) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .paint)
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Two layers of opposing-direction warp simulate trowel strokes.
                let stroke1 = Noise.fbmTiled(u, v, baseCells: 3, octaves: 4, seed: seed)
                let stroke2 = Noise.fbmTiled(v, u, baseCells: 3, octaves: 4, seed: seed ^ 0xA9)  // transpose → 90° grain
                let blend = 0.5 * (stroke1 + stroke2)
                // Macro luminosity variation. The `cloud` (single-cell) wash is the LOW-FREQUENCY
                // blotch that reads as grime on a flat wall (Danny, 2026-07-13) — cut it right down
                // so venetian reads as clean burnished plaster, not dirty stucco. The trowel STROKE
                // (`blend`, 3-cell directional) is the finish's identity and stays; a finer `micro`
                // (16-cell) tooth carries the tonal life the flat-colour gate needs at a scale that
                // reads as burnished texture, not a metre-wide dirt patch.
                let cloud = Noise.fbmTiled(u, v, baseCells: 1, octaves: 3, seed: seed ^ 0x55)
                let micro = Noise.fbmTiled(u, v, baseCells: 16, octaves: 2, seed: seed ^ 0xF1)
                // Base color: creamy off-white clamped well inside the dielectric band [0.04, 0.94].
                let lum = clamp01(0.74 + 0.10 * blend + 0.020 * cloud + 0.045 * micro)
                ch.albedo[ch.idx(x, y)] = Vec3(lum, lum * 0.985, lum * 0.97)   // very slight warm cast
                // Roughness: smooth where polished, rougher in trowel valleys. Range must
                // clear the roughnessFloor = 0.012 std-dev audit gate.
                ch.roughness[ch.idx(x, y)] = clamp01(0.18 + 0.22 * (1.0 - blend) + 0.06 * micro)
                // Height from the two stroke layers — drives the macro normal.
                ch.height[ch.idx(x, y)] = clamp01(0.50 + 0.28 * blend + 0.22 * cloud)
            }
        }
        ch.clearcoat = 0.25      // burnished polish lobe
        ch.deriveNormals(strength: 2.0)
        // Fine troweled micro-texture — the burnished-plaster tooth beneath the polish.
        // Subtle: burnished venetian plaster is smooth, just not glassy.
        addMicroDetail(&ch, seed: seed ^ 0xB5, baseCells: 80, strength: 0.35)
        return ch
    }

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
                let olive = 0.02 * (mottle - 0.5)   // ±0.01 low-freq yellow-olive patch tint on R
                let lush = Vec3(clamp01(0.075 + gLevel * 0.14 + olive),
                                clamp01(0.06 + gLevel * 0.40),
                                clamp01(0.045 + gLevel * 0.095))
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

    /// Clipped **boxwood** hedge foliage — a dense mass of small waxy leaves read at close
    /// range, NOT turf. The hedge placeable used the `grass` tile as a stand-in, but grass is
    /// soil-gapped vertical blades: it reads as lawn stood on end, never as a trimmed shrub.
    /// Boxwood is the opposite surface — a continuous canopy of overlapping oval leaves with
    /// dark recesses between the sprays, deep green shading to lighter yellow-green where fresh
    /// growth catches the sun, and a faint waxy leaf sheen. All structure is a Worley leaf
    /// lattice + tiled fbm undulation, so it is fully stochastic and tileable — no coherent
    /// period to band in a wide yard shot — and `.ground` category (yard-only + the hex
    /// de-repeat, exactly like `grass`).
    ///
    /// The leaf relief lives ENTIRELY in the material (the hedge mesh is a clipped box, no
    /// per-leaf geometry), so this is a `.needsMicroRelief` surface — it ships a detail normal.
    public static func boxwood(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 47) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .ground)
        // Deep boxwood green, and the lighter yellow-green of freshly clipped new growth.
        let deep  = Vec3(0.045, 0.115, 0.040)
        let fresh = Vec3(0.150, 0.250, 0.075)
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
                let coarse = Noise.voronoiTiled(wu, wv, cells: 11, jitter: 0.95, seed: seed ^ 0x5C2D)
                let fine   = Noise.voronoiTiled(wu, wv, cells: 22, jitter: 0.95, seed: seed ^ 0x77E9)
                let dome  = clamp01(1.0 - fine.f1 * 1.4)                    // 1 at a leaf centre
                let broad = clamp01(1.0 - coarse.f1 * 1.2)                  // 1 at a spray centre
                // Per-leaf tint, biased toward its broad cluster so whole sprays vary together.
                let leafT = mix(Noise.unit(fine.cellId), Noise.unit(coarse.cellId), 0.4)
                // Within-leaf micro tonal variation (waxy highlight vs. shaded lamina).
                let micro = Noise.fbmTiled(u * 6.0, v * 6.0, baseCells: 8, octaves: 2, seed: seed ^ 0x91A3)

                // A fraction of leaves are fresh yellow-green new growth, biased toward the
                // sun-lit sprays; the rest stay deep green.
                let freshFrac = clamp01((leafT - 0.50) * 1.6) * clamp01(0.4 + spray)
                var col = mix(deep, fresh, freshFrac)
                // BROKEN gap: the leaf-edge outline only darkens where a micro-noise agrees, so
                // the shadow between leaves reads as intermittent pockets, not a crackle net.
                let edge = 1.0 - smoothstep(0.0, 0.10, fine.f2 - fine.f1)
                let pocket = edge * smoothstep(0.35, 0.70, micro)
                let shade = clamp01(0.26 * (1.0 - spray) + 0.42 * pocket)   // 0 lit … 1 recess
                col = col * (0.82 + 0.30 * micro) * (1.0 - 0.50 * shade)
                ch.albedo[ch.idx(x, y)] = clampBand(col)

                // Waxy leaf faces read semi-glossy; the shaded recesses go matte. The spatial
                // swing carries the roughness-std TextureAudit tell.
                ch.roughness[ch.idx(x, y)] = clamp01(0.52 + 0.30 * shade + 0.08 * (micro - 0.5))

                // Leaf clusters stand proud, recesses sink — the macro relief that reads as
                // many small overlapping leaves once deriveNormals runs.
                ch.height[ch.idx(x, y)] = clamp01(0.42 + 0.30 * dome + 0.18 * broad
                                                  + 0.14 * spray - 0.22 * pocket)
            }
        }
        ch.clearcoat = 0.08          // faint waxy leaf sheen — not a wet gloss
        ch.deriveNormals(strength: 2.6)
        // Fine leaf-lamina tooth (the macro leaves are in the height field above; this is the
        // sub-leaf grain that stops the canopy reading plastic at a grazing angle).
        addMicroDetail(&ch, seed: seed ^ 0xB6, baseCells: 110, strength: 0.40)
        return ch
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
                let cleanLum  = clamp01(0.72 + 0.04 * broom)
                let stainLum  = clamp01(0.48 + 0.06 * weather)
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
                    mortarColor: Vec3 = Vec3(0.72, 0.70, 0.66),
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
    /// `grainAspect` stretches every feature ALONG V by that factor — the anisotropic micro-relief
    /// a directional material needs (wood's pore channels run down the board; a round pit is what
    /// made oak read as corrugated). It multiplies the U frequency rather than dividing V's so the
    /// coarsest octave keeps its cell count, and it must stay an INTEGER: `fbmTiled` wraps on a
    /// `cells`-periodic lattice, so a fractional multiplier puts a discontinuity at the UV wrap.
    static func addMicroDetail(_ ch: inout MaterialChannels, seed: UInt64,
                               baseCells: Int = 96, octaves: Int = 2, strength: Double = 0.9,
                               occlusionStrength: Double? = nil, grainAspect: Int = 1) {
        guard ch.detailNormal == nil else { return }
        let n = ch.size
        let aspect = Double(max(1, grainAspect))
        var h = [Double](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                let u = Double(x) / Double(n), v = Double(y) / Double(n)
                h[ch.idx(x, y)] = Noise.fbmTiled(u * aspect, v, baseCells: baseCells,
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
                             base: Vec3 = Vec3(0.90, 0.86, 0.78)) -> MaterialChannels {
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
                let shade = 0.955 + 0.05 * crown
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * slub * shade)
                // Roughness is high and EVEN: linen is a very matte cloth, so keep the mean up
                // (~0.90). The variation is LOW-FREQUENCY (broad smooth patches, NOT fine per-texel
                // speckle and NOT the periodic weave) so it gives subtle fabric life without a grain
                // that reads as a dirty mottle on a dimly-lit panel.
                ch.roughness[ch.idx(x, y)] = clamp01(0.90 - crown * 0.02
                    + (Noise.fbmTiled(u * 2.5, v * 2.5, baseCells: 2, octaves: 2, seed: sh ^ 0x3F) - 0.5) * 0.18)
                ch.height[ch.idx(x, y)] = clamp01(0.40 + 0.34 * crown * slub + 0.05 * fuzz)
            }
        }
        // A MODERATE cloth sheen — the soft directional nap glow real linen has. Kept well below
        // velvet grade so it reads as matte upholstery, not satin: the fabric character is carried
        // mostly by the albedo weave + roughness variation, with sheen as the finishing soft glow.
        ch.sheen = 0.30
        ch.grainTangent = [Vec2](repeating: Vec2(1, 0), count: size * size)
        ch.deriveNormals(strength: 0.1)            // near-flat — the weave is only a hint, so a
                                                   // grazing back cushion stays even soft cloth,
                                                   // never a specular mesh/screen grid
        // Fine sub-thread fuzz — the linen tooth at grazing range, kept to a whisper (a stronger
        // detail normal sparkles into a dirty grain on a dimly-lit grazing panel).
        //
        // S2.4: that 0.03 is a limit on the NORMAL, not a statement that linen's weave is
        // 3 % deep — and the occlusion has no sparkle failure mode, so it takes the weave's
        // real depth. Measured at the grazing gate: 1.01 (normal-only) → 1.04 inheriting 0.03,
        // → 1.2× at the weave's own strength, which is what a linen cushion should do.
        addMicroDetail(&ch, seed: seed ^ 0xBC, baseCells: 100, strength: 0.03, occlusionStrength: 0.6)
        return ch
    }

    /// Brushed nickel — warm silver fixture finish: fine linear grain, thin lacquer coat.
    public static func brushedNickel(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 113) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        let sh = seed
        let base = Vec3(0.72, 0.70, 0.67)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let grain = Noise.fbmTiled(u * 64, v * 2, baseCells: 8, octaves: 3, seed: sh)
                let macro = 0.92 + 0.16 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 2, seed: sh ^ 0x1F)
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * (0.88 + 0.24 * grain))
                ch.roughness[ch.idx(x, y)] = clamp01(0.22 + (grain - 0.5) * 0.18
                    + (Noise.fbmTiled(u, v, baseCells: 50, octaves: 2, seed: sh ^ 0x2F) - 0.5) * 0.04)
                ch.height[ch.idx(x, y)] = clamp01(0.50 + (grain - 0.5) * 0.08)
            }
        }
        ch.grainTangent = [Vec2](repeating: Vec2(1, 0), count: size * size)
        ch.clearcoat = 0.20
        ch.deriveNormals(strength: 3)
        // Fine satin tooth between the grain lines — a brushed-nickel fixture isn't a mirror.
        addMicroDetail(&ch, seed: seed ^ 0xBD, baseCells: 100, strength: 0.30)
        return ch
    }

    /// The colourway a bare `matteBlack()` bakes — the near-neutral powder-coat black, luma 0.10
    /// with the historical faint-warm tint (×0.98 green, ×0.96 blue). Declared once so the
    /// registry's `matte-black` default and the generator's default `color:` cannot drift apart.
    public static let matteBlackDefaultColor = Vec3(0.10, 0.098, 0.096)

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
                let coat  = Noise.fbmTiled(u * 3, v * 3, baseCells: 5, octaves: 3, seed: sh ^ 0xAA)
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
        let base = Vec3(0.90, 0.89, 0.88)
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
                let poreDark = smoothstep(0.0, 0.12, pore.f2 - pore.f1)
                // Pull-up: where creases occur (low macro noise), leather lightens
                let pullUp = smoothstep(0.6, 0.9, macro) * 0.12
                let poreDepth = (1 - poreDark) * 0.08
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * (0.88 + poreDark * 0.12 + pullUp))
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
        addMicroDetail(&ch, seed: seed ^ 0xBF, baseCells: 90, strength: 0.55)
        return ch
    }

    /// Aluminum — anodized brushed finish: warm grey with fine directional grain,
    /// slightly rougher than chrome, used for window frames and modern fixtures.
    public static func aluminum(size: Int = MaterialGenerator.bakeSize, seed: UInt64 = 131) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .metal)
        let sh = seed
        let base = Vec3(0.77, 0.77, 0.76)
        for y in 0..<size {
            for x in 0..<size {
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let grain = Noise.fbmTiled(u * 40, v * 3, baseCells: 8, octaves: 3, seed: sh)
                let macro = 0.90 + 0.20 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 2, seed: sh ^ 0x3F)
                let scratch = Noise.fbmTiled(u * 60, v * 2, baseCells: 8, octaves: 2, seed: sh ^ 0x6D)
                ch.albedo[ch.idx(x, y)] = clampBand(base * macro * (0.85 + 0.30 * grain + 0.10 * scratch))
                ch.roughness[ch.idx(x, y)] = clamp01(0.32 + (grain - 0.5) * 0.20
                    + (scratch - 0.5) * 0.12)
                ch.height[ch.idx(x, y)] = clamp01(0.50 + (grain - 0.5) * 0.06)
            }
        }
        ch.grainTangent = [Vec2](repeating: Vec2(1, 0), count: size * size)
        ch.clearcoat = 0.10
        ch.deriveNormals(strength: 2)
        // Fine satin tooth — anodized/brushed aluminum has a between-grain micro-structure,
        // not a mirror finish. Subtle, so the directional grain still dominates.
        addMicroDetail(&ch, seed: seed ^ 0xC0, baseCells: 100, strength: 0.30)
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
