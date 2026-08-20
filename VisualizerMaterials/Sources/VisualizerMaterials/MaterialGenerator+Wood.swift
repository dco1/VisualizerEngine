import simd
import Foundation

// MARK: - What kind of wood

/// **The species a `Wood` material is cut from.**
///
/// One picker entry ("Wood") with a species choice, rather than one library entry per tree —
/// the same consolidation `Tile` got when the subway/square split collapsed into `TileParams`
/// ([[feedback-unify-sibling-entities]]). Adding a wood is a `WoodRecipe` here, not a registry
/// row, a generator, a swatch and a verify script.
///
/// The recipe below is **wood anatomy, not a look**: the species we ship differ in the way the
/// trees actually differ — oak is ring-porous (a band of large vessels opens each growth ring,
/// which is why its grain has visible pore channels you can feel), cherry is diffuse-porous
/// (vessels are small and evenly spread, so the surface is closed and glassy, and its tell is
/// the scattered gum fleck rather than the pore), and pine is not a hardwood at all — a conifer
/// has no vessels to be porous WITH, so it has neither an open pore nor a ray fleck, and what
/// identifies it instead is the violent density step from its pale earlywood to its hard dark
/// latewood, and its knots.
public enum WoodSpecies: String, CaseIterable, Codable, Sendable, Identifiable {
    /// American white oak — the hardwood floor.
    case oak
    /// American black cherry — the cabinet wood.
    case cherry
    /// Eastern white / southern yellow pine — the knotty softwood.
    case pine

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .oak:    return "Oak"
        case .cherry: return "Cherry"
        case .pine:   return "Pine"
        }
    }

    /// One line for the picker — what makes this wood look like itself.
    public var summary: String {
        switch self {
        case .oak:    return "Ring-porous — cathedral figure, open pore channels, honey brown."
        case .cherry: return "Diffuse-porous — fine closed grain, warm red-brown, gum flecks."
        case .pine:   return "Softwood — pale cream, hard dark latewood bands, knotty."
        }
    }

    /// A stable per-species bake seed, so two species never draw the same board-tone sequence.
    /// `oak` keeps 3 — the seed the flooring oak has always baked at.
    var seed: UInt64 {
        switch self {
        case .oak:    return 3
        case .cherry: return 23
        case .pine:   return 41
        }
    }

    var recipe: WoodRecipe {
        switch self {
        case .oak:    return .whiteOak
        case .cherry: return .blackCherry
        case .pine:   return .knottyPine
        }
    }
}

/// **How the wood is laid up on the surface.**
///
/// Not decoration: a floor is boards with side seams and staggered butt joints, a case good is
/// one continuous veneered face. Baking flooring onto a dresser front is what made the drawer
/// fronts read as chipboard cross-hatch (realism audit S4), and the fix belongs in the material,
/// not in a second library entry.
public enum WoodLayout: String, CaseIterable, Codable, Sendable, Identifiable {
    /// Flooring / decking / laminations — side-seam grooves plus staggered butt joints.
    case boards
    /// One continuous face — furniture veneer, a stair's stepped solid, a panel.
    case panel

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .boards: return "Boards"
        case .panel:  return "Panel"
        }
    }
}

/// **A wood material's full customization** — species, lay-up, and how wide a board is.
///
/// `boardWidthInches` is a REAL WORLD SIZE, and the surface's UV period is then taken FROM it —
/// the same inversion tile and terrazzo use, and for the same reason (see `targetRunMeters`).
/// The flooring oak used to hardcode `planks: 16`, which is only "5 inches" if the surface
/// happens to tile at exactly 2 m; now the board is 5″ because 5″ is what it says.
public struct WoodParams: Equatable, Hashable, Sendable, Codable {
    public var species: WoodSpecies
    public var layout: WoodLayout
    /// Face width of one board, inches. Clamped to a millable 2″…12″.
    public var boardWidthInches: Double
    /// **Knots.** On by default, because a board without one is a board nobody milled: clear
    /// vertical-grain stock is graded, expensive and rare, and its absence is one of the things
    /// that reads as "printed" rather than "sawn". Off gives select/clear grade.
    public var knots: Bool

    public init(species: WoodSpecies = .oak,
                layout: WoodLayout = .boards,
                boardWidthInches: Double = 5.0,
                knots: Bool = true) {
        self.species = species
        self.layout = layout
        self.boardWidthInches = Swift.max(2, Swift.min(12, boardWidthInches))
        self.knots = knots
    }

    /// Tolerant decode — a document written before a field existed still loads.
    public init(from decoder: Decoder) throws {
        let d = try decoder.container(keyedBy: CodingKeys.self)
        let def = WoodParams()
        self.init(species: try d.decodeIfPresent(WoodSpecies.self, forKey: .species) ?? def.species,
                  layout: try d.decodeIfPresent(WoodLayout.self, forKey: .layout) ?? def.layout,
                  boardWidthInches: try d.decodeIfPresent(Double.self, forKey: .boardWidthInches)
                      ?? def.boardWidthInches,
                  knots: try d.decodeIfPresent(Bool.self, forKey: .knots) ?? def.knots)
    }

    public var boardWidthMeters: Double { boardWidthInches * 0.0254 }

    /// **How wide a wood repeat wants to be, in metres — and why it is not the surface's 2 m.**
    ///
    /// A GPU atlas slice is 512², fixed. A bake spread over a room floor's nominal 2 m is
    /// therefore 3.9 mm per texel, which gives a 5″ board **32 texels**; oak grain at a real
    /// 8–12 mm ring spacing needs 13–16 cycles in there, i.e. two texels each. That is below
    /// Nyquist, so the lines cannot be drawn — they can only be aliased, which is why the
    /// flooring oak read as flat-with-shimmer no matter how its colours were tuned. The two
    /// materials Danny has called photoreal — tile and terrazzo — are the two whose generators
    /// pick their own run for exactly this reason ([[daydream-material-scale-auditor]]).
    ///
    /// At ~1 m a board gets 64 texels and 2 mm per texel, which draws real ring spacing with
    /// texels to spare. The cost is that the tile repeats twice as often; that is paid for by
    /// keeping butt joints sparse (`WoodRecipe.jointDensity`) so the repeat carries no strong
    /// landmark, and by the per-board tone de-repeat (`patternCells`), which re-tones every
    /// PHYSICAL board column across the floor rather than every tile.
    public static let targetRunMeters = 1.0

    /// World metres one repeat spans on `tiling`.
    ///
    /// On an **adjustable** surface (walls, room floors and ceilings — the mesh is converted with
    /// the resolved material in hand) the wood chooses: a whole number of boards nearest the
    /// target, so a repeat is always a board boundary and the wrap lands under a seam groove.
    /// On a **prebaked** surface the UVs were baked earlier against a dummy material and nothing
    /// chosen here can move them, so the wood reads the truth instead of rendering a size it was
    /// never going to get.
    public func run(on tiling: SurfaceTiling) -> Double {
        guard tiling.isAdjustable else { return tiling.metresPerRepeat }
        guard layout == .boards else { return Self.targetRunMeters }
        let n = Swift.max(1, Int((Self.targetRunMeters / boardWidthMeters).rounded()))
        return Double(n) * boardWidthMeters
    }

    /// Boards across one repeat. A `panel` is one board by definition, and a surface whose repeat
    /// is narrower than a board gets one too — a dresser side is not two half-boards, it is a
    /// face.
    public func planks(on tiling: SurfaceTiling) -> Int {
        guard layout == .boards else { return 1 }
        return Swift.max(1, Int((run(on: tiling) / boardWidthMeters).rounded()))
    }

    /// True when this lay-up actually draws seams on `tiling` — one board across a repeat has no
    /// side seam to draw and no second board to stagger a butt joint against.
    public func hasSeams(on tiling: SurfaceTiling) -> Bool { planks(on: tiling) > 1 }
}

// MARK: - Species anatomy

/// **The per-species dials the shared grain engine reads.** Everything here is a statement about
/// the wood; the *structure* (grain along the board, growth rings across it, along-grain pores,
/// cross-grain ray fleck, per-board tone) is constant and lives in `woodGrain`.
struct WoodRecipe: Sendable {
    /// The light spring-growth colour and the darker summer band it blends toward, linear.
    var earlywood: Vec3
    var latewood: Vec3

    /// **Growth rings per METRE of board face** — the one physical number ring spacing comes
    /// from. Expressed per metre, not "per board", so the same wood draws the same grain
    /// whether it is a 5″ floor board or a 1 m panel; the old per-board count made a veneer
    /// silently 16× coarser than the floor it was cut from.
    var ringsPerMeter: Double
    /// How irregular ring spacing is. 0 draws a metronome (the tell that reads as printed
    /// laminate); real rings vary with the season that grew them.
    var ringJitter: Double
    /// Narrowness of the dark latewood band — the `pow` exponent on the ring profile. Lower is
    /// a broader, softer band.
    var latewoodSharpness: Double
    /// Cathedral-arch warp: how far the grain lines bow along the board. 0 is quarter-sawn
    /// straight, ~1 is the pronounced flat-sawn arch.
    var distort: Double

    /// Open-grain pore strength. 1 is oak's ring-porous vessel channels; a diffuse-porous wood
    /// (cherry, maple) is near-closed and sits an order down.
    var pore: Double
    /// Medullary ray fleck — the short CROSS-grain dashes of quarter-sawn oak.
    var ray: Double
    /// Gum pockets / pith flecks — small dark specks, cherry's signature.
    var gum: Double
    /// **Knots per square metre of surface.** Handed to the RENDERER rather than drawn into the
    /// tile — see `MaterialChannels.knots`. `WoodParams.knots` switches the feature off.
    var knotsPerSquareMeter: Double
    /// Mean knot radius in METRES — a tight flooring knot, not a timber-frame one.
    var knotRadius: Double
    /// How dark the knot core is against the wood it grew through.
    var knotDarkness: Double

    /// Per-board tone spread, ± of the board's value. Boards are cut from different trees.
    var boardTone: Double
    /// Chance a board carries a butt joint in a repeat — the average board length, and the one
    /// number that decides how loudly the tile announces itself. A joint on every board every
    /// repeat is a rung ladder at the repeat pitch, which is not what a floor looks like; at 0.35
    /// on a ~1 m repeat the mean board runs ~2.9 m and most repeats show no joint at all.
    var jointDensity: Double

    /// Base roughness of the FINISH (these woods are sealed), and how much the grain moves it.
    /// Held tight on purpose: a wide roughness swing paints patchy gloss, and patchy gloss on a
    /// floor does not read as grain, it reads as standing water.
    var roughBase: Double
    var roughSpread: Double
    var clearcoat: Double

    /// Amplitude of the MACRO normal relief (seams, joints, a whisper of ring).
    var relief: Double

    /// The DETAIL band — the only channel with the resolution to draw a pore. `cells` is the
    /// base lattice across the detail tile; `aspect` stretches each feature ALONG the grain
    /// (integer, so the field still wraps).
    var microCells: Int
    var microAspect: Int
    var microStrength: Double

    /// **American white oak, finished.** The library's flooring wood; its albedo is unchanged
    /// from the bake Danny called "almost there", because the defect was never the colour.
    static let whiteOak = WoodRecipe(
        earlywood: Vec3(0.55, 0.40, 0.23),
        latewood:  Vec3(0.42, 0.300, 0.165),
        ringsPerMeter: 105, ringJitter: 0.55, latewoodSharpness: 2.2, distort: 0.30,
        pore: 1.0, ray: 0.5, gum: 0,
        knotsPerSquareMeter: 0.5, knotRadius: 0.023, knotDarkness: 0.60,
        boardTone: 0.11, jointDensity: 0.35,
        roughBase: 0.40, roughSpread: 0.10, clearcoat: 0.10,
        relief: 1.6,
        microCells: 14, microAspect: 8, microStrength: 0.55)

    /// **American black cherry, finished.** Diffuse-porous: the pores are invisible, the rings
    /// are close and low-contrast, and what identifies the wood is its warm red-brown and the
    /// scattered dark gum flecks. Takes a smoother finish than oak because there is no open
    /// pore for the film to sink into.
    static let blackCherry = WoodRecipe(
        earlywood: Vec3(0.310, 0.105, 0.055),
        latewood:  Vec3(0.245, 0.079, 0.040),
        ringsPerMeter: 135, ringJitter: 0.35, latewoodSharpness: 1.8, distort: 0.18,
        pore: 0.18, ray: 0.10, gum: 0.55,
        // Cherry throws small tight PIN knots rather than oak's round ones.
        knotsPerSquareMeter: 0.35, knotRadius: 0.015, knotDarkness: 0.62,
        boardTone: 0.08, jointDensity: 0.35,
        roughBase: 0.34, roughSpread: 0.06, clearcoat: 0.16,
        relief: 1.0,
        microCells: 40, microAspect: 3, microStrength: 0.22)

    /// **Knotty pine, finished.** The softwood, and it is the odd one out in every field that
    /// matters — which is the point of keeping this table anatomical rather than a set of tints.
    ///
    /// * **The early/latewood step is the whole look.** A conifer's earlywood is thin-walled and
    ///   almost white; its latewood is dense, resin-rich and much darker, and the change between
    ///   them is abrupt rather than graded. That is a 4.3:1 luminance ratio here against oak's
    ///   1.33:1, drawn narrow by a `latewoodSharpness` well past oak's — so the mean stays pale
    ///   and the bands read as hard lines rather than as a muddy overall darkening. Pine is the
    ///   one species where loud figure is correct.
    /// * **Wide, irregular rings — but STRAIGHT ones.** Plantation-grown softwood puts on
    ///   11–13 mm a year against oak's 9–10, so `ringsPerMeter` goes DOWN and `ringJitter` up.
    ///   88/m over a 127 mm board is 11.2 cycles against the bake's 14.2 ceiling
    ///   (`woodRingTexelFloor`), and 12.9 at the top of the per-board scale jitter — so the
    ///   number is live rather than silently band-limited, which cherry's 135/m is not at any
    ///   board width.
    /// * **`distort` is the LOWEST in the library, and that is a correction, not a taste.**
    ///   The bow is written in RING WIDTHS and divided by the cycle count, so the same `distort`
    ///   buys a wider swing on a wood with fewer rings: at 0.32 pine's lines wandered sideways
    ///   across 0.67 of a board — more than oak (0.43) and three times cherry (0.20) — and a
    ///   line that crosses two-thirds of its own board is not figure. Danny, 2026-08-19, on the
    ///   first pine render: *"it looks like Acrylic Pouring/Swirling/Marbling."* 0.10 puts the
    ///   swing at 0.17 of a board. Pine wants this: its rings are wide and its contrast is loud,
    ///   so every millimetre of wander is visible, and a knotty pine board really does read as
    ///   long parallel lines interrupted by knots rather than as cathedral swirl.
    /// * **No pore and no ray.** A conifer has no vessels and no broad rays: there is nothing to
    ///   draw. `pore`/`ray` at ~0, and a detail band both quieter and far less elongated than
    ///   oak's — reaching for its `microAspect: 8` would be inventing an open grain the tree
    ///   does not have.
    /// * **Knots are the species.** See `knotsPerSquareMeter` below for why 2.4 and not more.
    ///
    /// Colour is cream and pale yellow going to soft honey — less red than cherry, less brown
    /// than oak, and deliberately not the near-black latewood a literal reading of "very high
    /// contrast" produces: sooty bands on cream is charred pine, and the tonemapper turns any
    /// red-heavy dark band to caramel on the way to the screen. 4.3:1 in luminance against oak's
    /// 1.33:1 is still the loudest figure in the library. The latewood's blue is held at 0.062
    /// because `clampBand`'s 0.045 floor would lift anything lower and grey the bands anyway —
    /// the number says what the bake will actually contain.
    static let knottyPine = WoodRecipe(
        earlywood: Vec3(0.715, 0.578, 0.385),
        latewood:  Vec3(0.235, 0.115, 0.062),
        ringsPerMeter: 88, ringJitter: 0.72, latewoodSharpness: 3.6, distort: 0.10,
        // A conifer has no vessels and no broad rays. Not "subtle" — absent.
        pore: 0.05, ray: 0, gum: 0,
        // **2.4/m² is the lattice's ceiling, not pine's.** A #2-grade board really does throw a
        // knot every 300–600 mm, which is ~16/m²; `WoodKnotField.cellMeters` (0.5 m) holds one
        // candidate site per cell, so the hard cap is 4/m² and anything approaching it stops
        // being a scatter at all — at `cellDensity` 0.75 nearly every cell is occupied and the
        // field degenerates into a jittered GRID, which is the regular-looking failure Danny
        // already rejected once ("it looks like a pattern rather than organic"). 2.4/m² is
        // `cellDensity` 0.60: 4.8x oak's population, and 40% of cells still empty, which is what
        // leaves the clumps and gaps that read as organic. Going denser is a change to the
        // LATTICE, not to this number — see the note in `WoodKnotField.cellMeters`.
        knotsPerSquareMeter: 2.4, knotRadius: 0.030, knotDarkness: 0.80,
        boardTone: 0.14, jointDensity: 0.35,
        // Softwood takes finish unevenly — the soft earlywood drinks it, the hard latewood does
        // not — so the gloss genuinely does track the ring here. It is still narrow: the swing is
        // tied to `lw`, which runs ALONG the board, so it draws grain and not the blotch that
        // reads as standing water.
        roughBase: 0.44, roughSpread: 0.12, clearcoat: 0.07,
        relief: 1.5,
        microCells: 30, microAspect: 4, microStrength: 0.18)

    /// **Hard maple**, for the butcher-block counter's laminations. Not a picker species — a
    /// counter is not a finish you choose a tree for — but the same anatomy table, so it can't
    /// drift into being a second way of describing wood.
    static let hardMaple = WoodRecipe(
        earlywood: Vec3(0.60, 0.48, 0.32),
        latewood:  Vec3(0.52, 0.40, 0.26),
        ringsPerMeter: 110, ringJitter: 0.30, latewoodSharpness: 1.9, distort: 0.10,
        pore: 0.30, ray: 0.15, gum: 0,
        // A butcher block is edge-grain laminated from clear rips; a knot would be cut out.
        knotsPerSquareMeter: 0.12, knotRadius: 0.013, knotDarkness: 0.52,
        boardTone: 0.11, jointDensity: 0,
        roughBase: 0.42, roughSpread: 0.08, clearcoat: 0.08,
        relief: 1.4,
        microCells: 34, microAspect: 4, microStrength: 0.35)

    /// The sealed / grain-FILLED lay-up of the same wood. Furniture veneer is filled and
    /// lacquered, so its open pore is a whisper: left at flooring strength the pore relief packs
    /// into a row of hard bright/dark "stitching ticks" on the thin grazing-lit strips of a
    /// shaker face (the flat sides never graze, so they stayed smooth). Colour and ring anatomy
    /// are untouched — it is the same tree.
    var sealed: WoodRecipe {
        var r = self
        r.pore *= 0.35
        r.relief *= 0.45
        r.microStrength *= 0.35
        r.clearcoat = Swift.max(clearcoat, 0.14)
        // Case goods are plain- or quarter-sliced, which is a straighter figure than the
        // flat-sawn floor board the same tree gives. Left at flooring strength, a 1 m panel spans
        // several rings of cathedral swing and a table top reads as rotary-cut burl.
        r.distort *= 0.55
        // Case-good veneer is SELECTED stock — a knot on a drawer front is a reject, which is
        // also why the UI does not offer the knot control for a panel.
        r.knotsPerSquareMeter = 0
        return r
    }
}

// MARK: - The generators

/// Directional wood-grain generators (MATERIALS_AND_TEXTURES §5a).
///
/// The model:
///
///   • **Long directional grain LINES run along the board (V axis).** Growth rings stack ACROSS
///     the board width (U), so the latewood bands draw continuous streaks DOWN the board. A
///     low-frequency drift bows them into the cathedral arches of flat-sawn stock.
///   • **Ring spacing is physical and irregular** — `ringsPerMeter` on the recipe, warped by a
///     per-board jitter so the lines are not a metronome.
///   • **Pores are elongated ALONG the grain**, and the fine ones live in the DETAIL band, which
///     is the only channel with the resolution to draw them (see the anisotropy note below).
///   • **Medullary ray flecks run ACROSS the grain** — that is what a ray is.
///   • Roughness varies, but narrowly: these woods are sealed, and a wide gloss swing reads as
///     water, not as figure.
///
/// ## The anisotropy convention, because it was inverted here for a year
///
/// `Noise.fbmTiled(u · K, v · L, baseCells: C)` lays a `K·C × L·C` lattice over the tile, so a
/// feature is `1/(K·C)` wide and `1/(L·C)` tall. **A feature elongated ALONG the board (down V)
/// therefore needs K ≫ L** — many cycles across U, few down V.
///
/// Every field here used to be written the other way round: the pore field was
/// `fbmTiled(u·6, v·48)` under a comment claiming "high V-frequency … fine vertical streaks",
/// which actually draws features eight times WIDER than tall — cross-grain rungs. They were
/// written into height (→ normals) and roughness as well as albedo, so every board carried a
/// stack of horizontal ridges with patchy gloss between them: Danny, 2026-08-19, *"the
/// horizontal bumps that go across … make it look like it's been flooded or waterlogged."*
/// The ray field carried the mirror-image mistake (drawn along the grain, which is the one
/// direction a medullary ray never runs), and the macro tone field was isotropic, which puts
/// round dark blotches on a floor — the shape of a water stain.
///
/// ## Why the pore moved to the detail band
///
/// A 512² bake over a 2 m floor is 3.9 mm per texel. An oak pore is 0.2–0.4 mm: it is not
/// merely absent from the macro channels, it is **unrepresentable** there, and drawing it
/// anyway costs a 1–2 texel feature in the height map, which is normal-map noise rather than
/// texture ([[daydream-material-scale-auditor]]). The `detailNormal`/`detailOcclusion` band is
/// sampled at `detailNormalUVScale ×` UV — 0.25 m over 512 texels, 0.49 mm per texel — so that
/// is where the pore channels are drawn, stretched along the grain by `microAspect`. The macro
/// height keeps only what it can resolve: the seam grooves, the butt joints, and a whisper of
/// ring relief.
extension MaterialGenerator {

    /// **The finest a growth ring may be drawn, in texels per cycle.** A cycle narrower than
    /// this is not a grain line — it is moiré, and it enters the mip chain as shimmer.
    ///
    /// The flooring oak used to ask for 11 rings across a 32-texel board: 2.9 texels per
    /// light-dark pair, with the sub-ring harmonic at 1.3. That is below Nyquist, which is why
    /// no amount of colour tuning ever made the grain read as grain. `ringsPerMeter` is
    /// band-limited against this rather than being quietly wrong.
    public static let woodRingTexelFloor = 4.5

    /// **Wood** — the one entry point. Species, lay-up and board width in, channels out.
    ///
    /// `tiling` says what the surface can offer. On an adjustable surface the wood picks the
    /// period from its board width (`WoodParams.run(on:)`) — callers on the render path must take
    /// that same number back as the mesh's UV scale, or the boards render at a width nobody
    /// chose. On a prebaked surface the wood adapts to the period the mesh already has.
    public static func wood(size: Int = MaterialGenerator.bakeSize,
                            params: WoodParams = WoodParams(),
                            tiling: SurfaceTiling = .librarySwatch) -> MaterialChannels {
        let planks = params.planks(on: tiling)
        let seams = params.hasSeams(on: tiling)
        var recipe = seams ? params.species.recipe : params.species.recipe.sealed
        if !params.knots { recipe.knotsPerSquareMeter = 0 }
        return woodGrain(size: size, planks: planks, seed: params.species.seed,
                         recipe: recipe, runMeters: params.run(on: tiling),
                         category: .wood, seams: seams, endJoints: seams)
    }

    /// Oak at an explicit plank count and run — the direct handle on the shared engine, for
    /// tests and for the one caller that genuinely knows its own board count.
    public static func oak(size: Int = MaterialGenerator.bakeSize, planks: Int = 8,
                           runMeters: Double = WoodParams.targetRunMeters,
                           seed: UInt64 = 3) -> MaterialChannels {
        woodGrain(size: size, planks: planks, seed: seed, recipe: .whiteOak,
                  runMeters: runMeters,
                  category: .wood, seams: planks > 1, endJoints: planks > 1)
    }

    /// FURNITURE-grade oak veneer: one continuous board face, sealed and grain-filled. Kept as a
    /// named generator because the render layer resolves stair treads and furniture slices by
    /// the `oak-veneer` id; it is `wood(layout: .panel)` and nothing else.
    public static func oakVeneer(size: Int = MaterialGenerator.bakeSize,
                                 tiling: SurfaceTiling = .librarySwatch) -> MaterialChannels {
        wood(size: size, params: WoodParams(species: .oak, layout: .panel), tiling: tiling)
    }

    /// The shared directional-grain engine behind every plank-style wood. `recipe` picks the
    /// species; the structure is constant and seamless by construction.
    ///
    /// - Parameters:
    ///   - runMeters: world metres one repeat spans. Ring spacing is physical, so the bake has
    ///     to know how big it is — this is the number that makes `ringsPerMeter` mean something.
    ///   - seams: emit plank side-seam grooves (flooring/laminations). FALSE for a continuous
    ///     panel face.
    ///   - endJoints: emit the staggered butt joints where boards END. Separate from `seams`
    ///     because a laminated COUNTER has continuous strips with glue lines down their length
    ///     and no butt joints across them.
    static func woodGrain(size: Int, planks rawPlanks: Int, seed: UInt64,
                          recipe r: WoodRecipe,
                          runMeters: Double = 2.0,
                          category: MaterialCategory = .wood,
                          seams: Bool = true,
                          endJoints: Bool = true) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: category)
        let planks = max(1, rawPlanks)

        // S2.5 half 2 — boards run unbroken down v with one tone per u-column (the `bh` hash
        // below is per board index), so the per-cell de-repeat is (planks, 0): one fresh tone per
        // PHYSICAL board column across the whole floor, none down a board. A seamless single
        // face gets none — one cell per UV tile would be exactly the 2 m-checkerboard failure
        // the field's contract warns about.
        if seams && planks > 1 {
            ch.patternCells = Vec2(Double(planks), 0)
            ch.patternJitter = 0.05
        }

        // ── Ring frequency: physical first, then band-limited to what the bake can draw. ──
        let boardMeters = runMeters / Double(planks)
        let boardTexels = Double(size) / Double(planks)
        let maxCycles = max(1.0, boardTexels / woodRingTexelFloor)
        let wantCycles = r.ringsPerMeter * boardMeters

        var grain = [Vec2](repeating: Vec2(0, 1), count: size * size)

        for y in 0..<size {
            for x in 0..<size {
                // texel centre → seam-safe sampling of the board boundary on both wrap columns
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)

                // ── Board index & local coordinate across the width (U) ──
                let bf = u * Double(planks)
                let board = Int(floor(bf))
                let bx = bf - floor(bf)                              // 0…1 across the board width
                let bh = Noise.hash2(board, 0, seed)
                // A per-board seed for the figure itself. Sharing one field across the whole tile
                // drew a single continuous wave that swept through every board at once, so the
                // boards read as one wide sheet scored with lines; a board is a different piece
                // of tree from its neighbour, and the seam is exactly where that may show.
                let bseed = Noise.hash2(board, 7, seed)

                let tone       = 1 - r.boardTone + 2 * r.boardTone * Noise.unit(bh)
                let scaleJit   = 0.85 + 0.30 * Noise.unit(bh ^ 0x55)
                let phase      = Noise.unit(bh ^ 0x9A) * 6.2832
                let hueWarm    = (Noise.unit(bh ^ 0x7C) - 0.5) * 0.06
                // Stagger butt joints so they never line up across boards.
                let vStag      = (v + Noise.unit(bh ^ 0x33)).truncatingRemainder(dividingBy: 1.0)

                // A SEAMLESS single face has no groove to hide the U-wrap, so its ring frequency
                // must be an INTEGER number of cycles across the tile or the grain jumps at the
                // wrap (TextureAudit's seam tell). Planked wood keeps the free frequency — each
                // board's wrap lands under a seam groove.
                let free = min(wantCycles * scaleJit, maxCycles)
                let cycles = seams ? free : max(1, free.rounded())

                // ── Cathedral bow: a SMOOTH, LOW-frequency lateral sweep of the whole ring
                // stack, so the lines nest into the arches of flat-sawn stock. (A 4-octave field
                // here reached ~32 cycles down the tile and read as liquid/melting grain.)
                //
                // The amplitude is written in RING WIDTHS, not in board fractions: a bow of
                // "0.045 of a board" means something different at every ring density, which is
                // why the old constant produced barely half a ring of swing at flooring scale and
                // no visible arch at all. `bowRings` says what it means. ──
                // **These two fields were the last isotropic ones in the generator, and being
                // isotropic is what made the grain SWIRL.** Danny, 2026-08-19, on the first pine
                // render: *"it looks like Acrylic Pouring/Swirling/Marbling."*
                //
                // The 2026-08-19 pass fixed the pore, ray and macro-tone fields to the K ≫ L
                // convention and left these alone, but they are the same defect: `drift` was
                // `fbmTiled(u, v, baseCells: 2)` — a 2×2 lattice, as tall as it is wide — and
                // `wiggle` was `fbmTiled(u·3, v·3, baseCells: 3)`, a 9×9 one. A field that
                // displaces the ring coordinate and varies as fast DOWN the board as across it
                // does not bow the grain, it makes it wander: `wiggle` put a lateral excursion
                // on a ~11 cm vertical period into every line, which is fluid movement, not
                // figure. A grain line has to be a LINE over its whole length; only where its
                // lateral position goes is up for grabs.
                //
                // So both are elongated along the board like everything else here. The vertical
                // lattice stays at 2 over the ~1 m tile — a half-metre sweep is a cathedral
                // arch, which is the one lateral movement flat-sawn stock really has — and the
                // cross-board frequency is what rises, which reshapes the arch without ever
                // bending a line along itself. Measured on pine (∂u/∂v of albedo luma, higher is
                // straighter): 3.99 → 12.31 on boards.
                let bowRings = r.distort * 12 * (0.4 + 1.2 * Noise.unit(bh ^ 0xC4))
                let drift  = (Noise.fbmTiled(u * 3, v, baseCells: 2, octaves: 2,
                                             seed: bseed ^ 0xAA) - 0.5) * 2 * bowRings / cycles
                let wiggle = (Noise.fbmTiled(u * 9, v, baseCells: 2, octaves: 2,
                                             seed: bseed ^ 0xBB) - 0.5) * 0.5 * bowRings / cycles

                // ── Ring-width irregularity. A warp of the cross-width coordinate, so it
                // compresses and opens the ring spacing the way a run of wet and dry seasons
                // does. Written in units of ONE RING (÷ cycles) so the same jitter reads the same
                // at any ring density, and built from whole cycles of `bx` so it still wraps. ──
                let jitterWave = sin(2 * .pi * bx + phase * 1.3) * 0.6
                               + sin(4 * .pi * bx + phase * 2.1) * 0.3
                let bxw = bx + r.ringJitter * jitterWave / cycles

                // ── Directional grain: rings across U ⇒ lines down V. `s` is the RING ORDINAL —
                // how many rings out from the pith this texel is — so a whole-number step of `s`
                // is one growth ring, and anything indexed by it is per-ring by construction. ──
                let s = (bxw + drift + wiggle) * cycles + phase / (2 * .pi)
                let g = sin(s * 2 * .pi)

                // **No two rings are alike, and that is most of what "not printed laminate"
                // means.** A 1-D value noise ON THE RING ORDINAL gives each ring its own darkness
                // and its own band width, while staying continuous (so it cannot draw a step
                // across the board where the bow carries `s` past a whole number — which a raw
                // `floor(s)` hash does, and which would be a horizontal artefact of exactly the
                // kind this pass exists to remove).
                let prominence = 0.45 + 0.95 * Noise.value1D(s, seed: seed ^ 0x2D)
                let sharpness  = r.latewoodSharpness * (0.65 + 0.75 * Noise.value1D(s + 37.5,
                                                                                    seed: seed ^ 0x3E))
                // Latewood (dark dense summer band): the troughs of the grain sinusoid.
                let ringT = 0.5 - 0.5 * cos((g * 0.5 + 0.5) * 2 * .pi)
                let lw = clamp01(pow(clamp01(ringT), sharpness) * prominence)

                // ── Open-grain pores: streaks ALONG the grain (many cycles across U, few down
                // V). In a RING-POROUS wood the big vessels open at the START of each ring, in
                // the earlywood — so the pore field is biased toward the light band, not the
                // dark one. ──
                let poreField = Noise.fbmTiled(u * 24, v * 3, baseCells: 4, octaves: 2,
                                               seed: seed ^ 0xCC)
                let pore = smoothstep(0.60, 0.80, poreField) * (0.40 + 0.60 * (1 - lw)) * r.pore

                // ── Medullary ray flecks: short dashes ACROSS the grain — a ray is a radial
                // ribbon, so it is the one feature of a board that runs the other way. ──
                let rayField = Noise.fbmTiled(u * 8, v * 32, baseCells: 4, octaves: 2,
                                              seed: seed ^ 0x5A)
                let fleck = smoothstep(0.76, 0.90, rayField) * r.ray

                // ── Gum pockets / pith flecks: small dark specks, slightly drawn out along the
                // grain. Cherry's signature; zero on the woods that don't have them. ──
                let gum = r.gum > 0
                    ? smoothstep(0.70, 0.84, Noise.fbmTiled(u * 20, v * 8, baseCells: 4,
                                                            octaves: 2, seed: seed ^ 0x6B)) * r.gum
                    : 0

                // ── Plank side seam: a thin recessed groove at each board edge. ──
                let edgeU = min(bx, 1 - bx)
                let seam = seams ? 1 - smoothstep(0.018, 0.030, edgeU) : 0

                // ── Butt joints. ONE candidate boundary per repeat, present only on some boards
                // (hashed per board) — a joint on every board every repeat is a rung ladder at
                // the repeat pitch, which is not what a floor looks like. At the default density
                // and a ~1 m repeat the mean board runs ~2.9 m, and most repeats show none. ──
                var joint = 0.0
                if seams && endJoints && r.jointDensity > 0,
                   Noise.unit(Noise.hash2(board, 1, seed ^ 0xE7)) < r.jointDensity {
                    // A butt joint is a HAIRLINE with a hard core, not a soft band: a wide
                    // gradient here reads as a shadow lying across the floor.
                    joint = 1 - smoothstep(0.0012, 0.0045, min(vStag, 1 - vStag))
                }

                // ── Colour. The macro tonal drift is STRETCHED ALONG THE GRAIN: an isotropic
                // field puts round dark patches on a floor, which is the shape of a water stain,
                // not of sapwood. ──
                let macro = 0.94 + 0.12 * Noise.fbmTiled(u * 6, v, baseCells: 2, octaves: 3,
                                                         seed: seed ^ 0xDD)
                var col = mix(r.earlywood, r.latewood, lw)
                col = Vec3(col.x * (1 + hueWarm), col.y, col.z * (1 - hueWarm))   // warm/cool board
                col = col * tone * macro
                col = col * (1 - 0.05 * pore)                                     // pores darken
                col = col * (1 - 0.30 * seam) * (1 - 0.22 * joint)                // grooves darken
                col = col * (1 + 0.07 * fleck)                                    // rays lighten
                col = col * (1 - 0.55 * gum)                                      // gum darkens
                ch.albedo[ch.idx(x, y)] = clampBand(col)

                // ── Roughness. Narrow on purpose (see `WoodRecipe.roughBase`): the finish is
                // what you are looking at, and the grain only modulates it. The break-up field
                // runs ALONG the grain like everything else. ──
                let rough = r.roughBase
                    + r.roughSpread * (lw - 0.5)
                    + 0.09 * pore + 0.14 * seam + 0.08 * joint - 0.03 * fleck
                    + (Noise.fbmTiled(u * 20, v * 2, baseCells: 4, octaves: 2,
                                      seed: seed ^ 0xEE) - 0.5) * 0.04
                ch.roughness[ch.idx(x, y)] = clamp01(rough)

                // ── Height: ONLY what a ~2 mm texel can resolve — the grooves, and a whisper of
                // ring relief. The pore lives in the detail band (see the type comment). ──
                ch.height[ch.idx(x, y)] = clamp01(0.5 + 0.06 * lw
                    - 0.50 * seam - 0.34 * joint + 0.03 * fleck)

                // Grain runs along the board (V), tilted slightly by the cathedral drift so the
                // anisotropic highlight follows the figure (TextureAudit tell #7).
                grain[ch.idx(x, y)] = norm(Vec2(0.10 * drift, 1))
            }
        }

        ch.grainTangent = grain
        // **Knots are declared, not drawn.** A knot is a sparse LANDMARK; baked into a ~1 m tile
        // it reappears on a 1 m grid across the whole floor, which is exactly what it looked like
        // (Danny, 2026-08-19: *"it looks like a pattern rather than organic"*). The renderer
        // evaluates them per pixel on the UNWRAPPED uv, which never repeats — see
        // `MaterialChannels.knots` and `sampleWoodKnots`.
        if r.knotsPerSquareMeter > 0 {
            ch.knots = WoodKnotField(perSquareMeter: r.knotsPerSquareMeter,
                                     radiusMeters: r.knotRadius,
                                     darkness: r.knotDarkness)
        }
        ch.clearcoat = r.clearcoat
        ch.deriveNormals(strength: r.relief)
        // The touch-scale pore band — stretched ALONG the grain, which is the whole point.
        addMicroDetail(&ch, seed: seed ^ 0xF1, baseCells: r.microCells,
                       strength: r.microStrength, grainAspect: r.microAspect)
        return ch
    }
}
