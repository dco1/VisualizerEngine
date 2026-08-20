import simd
import Foundation

// MARK: - The unit-pattern spine

/// **The shape of a laid unit**, as its long:short proportion. This is what used to be an
/// *identity* — `ceramic-tile` was square and `wall-tile` was subway, two near-duplicate
/// generators — and is now a *setting*, so one Tile material covers the whole family
/// (Danny, 2026-08-10: *"We need to be able to set the shape (square, rectangle, etc.)"*).
///
/// The aspect is an INTEGER on purpose. The bake is square in UV and spans `cols` units
/// across by `rows` down; for those to describe a physically square patch you need
/// `rows = cols × aspect`, and `rows` must be a whole number or the grid cannot wrap.
public enum UnitShape: String, Equatable, Hashable, Sendable, Codable, CaseIterable {
    /// 1:1 — a square floor tile, a mosaic chip.
    case square
    /// 2:1 — the true subway proportion (a 3″×6″ tile).
    case rectangle
    /// 3:1 — a plank tile / long-format wall tile.
    case plank
    /// 4:1 — a wood-look plank or a linear brick.
    case longPlank

    /// Long edge ÷ short edge.
    public var aspect: Int {
        switch self {
        case .square:    return 1
        case .rectangle: return 2
        case .plank:     return 3
        case .longPlank: return 4
        }
    }

    public var displayName: String {
        switch self {
        case .square:    return "Square"
        case .rectangle: return "Rectangle"
        case .plank:     return "Plank"
        case .longPlank: return "Long Plank"
        }
    }

    /// "6 × 3″" — the proportion spelled out at a concrete size, so the picker names a tile
    /// the way a tile shop does instead of naming a ratio.
    public func proportionLabel(longEdgeMeters m: Double) -> String {
        let long = m / 0.0254, short = long / Double(aspect)
        func f(_ v: Double) -> String {
            abs(v.rounded() - v) < 0.05 ? "\(Int(v.rounded()))" : String(format: "%.1f", v)
        }
        return aspect == 1 ? "\(f(long))″ square" : "\(f(long)) × \(f(short))″"
    }
}

/// **How courses are offset against each other.** Orthogonal to `UnitShape` — "subway tile"
/// is not a shape, it is `.rectangle` + `.running`, which is why these are two settings and
/// not one enum of named looks.
public enum UnitBond: String, Equatable, Hashable, Sendable, Codable, CaseIterable {
    /// Grid-aligned; every joint lines up. The classic square floor tile.
    case stack
    /// Each course shifted half a unit — brick/subway laying.
    case running
    /// Each course shifted a third — the "ashlar"/plank-floor look that avoids the
    /// ladder of aligned half-joints a running bond makes on long units.
    case thirdRunning

    /// Courses per offset cycle. **`rows` must be a multiple of this or the pattern cannot
    /// wrap**: the offset is a function of the row index, so at the v-seam row `rows` must
    /// land on the same phase as row 0. This is the running-bond tell that bit `unit-pavers`
    /// before it, and it is why `TileLayout` adjusts `cols` rather than trusting the ask.
    public var rowCycle: Int {
        switch self {
        case .stack:        return 1
        case .running:      return 2
        case .thirdRunning: return 3
        }
    }

    /// The u-shift (in units) applied to course `row`.
    public func offset(row: Int) -> Double {
        switch self {
        case .stack:        return 0
        case .running:      return (((row % 2) + 2) % 2 == 0) ? 0 : 0.5
        case .thirdRunning: return Double(((row % 3) + 3) % 3) / 3.0
        }
    }

    public var displayName: String {
        switch self {
        case .stack:        return "Stacked"
        case .running:      return "Running (½)"
        case .thirdRunning: return "Running (⅓)"
        }
    }
}

/// **Which way the units RUN.** A 3×6 subway laid long-edge-across is the classic backsplash; the
/// same tile stood on end is the stacked-vertical look. Same tile, same bond, different wall — so
/// it is a setting, not another shape (Danny, 2026-08-10: *"can the tile be set to run in different
/// patterns, like horizontal, vertical…?"*).
///
/// Implemented as an exact quarter turn of the whole lattice (the generator swaps `u`/`v` and the
/// grid counts), which on a square bake maps the pattern onto itself — so a vertical layout is
/// seamless by construction, exactly as the horizontal one is. **A 45° diagonal is deliberately
/// NOT offered here:** rotating a square lattice by 45° changes its period by √2, so the bake would
/// no longer wrap and every repeat would show a seam. That wants its own diamond-lattice generator,
/// not a rotation — flagged as a follow-up rather than shipped seamy.
public enum UnitRun: String, Equatable, Hashable, Sendable, Codable, CaseIterable {
    /// The unit's long edge runs across the surface — the default, and the only sensible one for
    /// a floor tile.
    case horizontal
    /// The unit's long edge runs up the surface: stacked/vertical subway, vertical planks.
    case vertical

    public var displayName: String { self == .horizontal ? "Horizontal" : "Vertical" }
}

/// **The surface sheen dial, for EVERY material** — not a tile-only control. Before this,
/// each generator hardcoded its own `clearcoat` and roughness band (tile 0.30, wall tile 0.35,
/// stoneware 0.12, sanitaryware 0.60), so "make this matte" had no expression anywhere.
public enum SurfaceFinish: String, Equatable, Hashable, Sendable, Codable, CaseIterable {
    case matte, satin, gloss

    /// The second-lobe clearcoat strength. Must stay in 0…1 (`TextureAudit.lobesInRange`).
    public var clearcoat: Double {
        switch self {
        case .matte: return 0.04
        case .satin: return 0.30
        case .gloss: return 0.55
        }
    }

    /// Base roughness of the unit BODY. The joint is always rougher — mortar/grout is
    /// unglazed by nature — so it is derived from this rather than dialled separately.
    public var bodyRoughness: Double {
        switch self {
        case .matte: return 0.52
        case .satin: return 0.24
        case .gloss: return 0.10
        }
    }

    public var displayName: String {
        switch self {
        case .matte: return "Matte"
        case .satin: return "Satin"
        case .gloss: return "Gloss"
        }
    }
}

/// **The joint/unit spine shared by tile, pavers, brick and plank.** Every one of those is
/// the same object: rectangular UNITS of a real-world size, laid in a BOND, separated by a
/// JOINT of some colour and width, with per-unit tone variation. `TileParams` and
/// `PaverParams` were literally this struct written twice (`PaverParams` already carried
/// `bodyColor`/`mortarColor`/`unitWidth`/`unitHeight`/`colorJitter`/`seed`), which is the
/// per-type fragmentation this codebase keeps retiring — so it is written once, here.
public struct UnitPatternParams: Equatable, Hashable, Sendable, Codable {
    public var shape: UnitShape
    /// The unit's **LONG edge in metres** — the number the user types. The short edge is
    /// derived as `unitSizeMeters / shape.aspect`, never authored separately, so a "6 inch
    /// subway tile" cannot drift into being 6 × 2.4″.
    public var unitSizeMeters: Double
    public var bond: UnitBond
    /// Which way the units run — see `UnitRun`. A no-op for `.square` (a square laid on end is the
    /// same square), so the UI only offers it when the shape has a long edge to point.
    public var run: UnitRun
    /// The unit face colour (linear, dielectric band).
    public var bodyColor: Vec3
    /// The grout/mortar colour. **Defaults to white** (Danny, 2026-08-10) — as white as the
    /// dielectric band allows, see `defaultJointColor`.
    public var jointColor: Vec3
    /// Joint width as a fraction of the unit's LONG edge, so the physical joint is equal on
    /// both axes (the v-axis half-width is scaled by the aspect at bake time). Kept
    /// proportional so a large-format tile keeps a proportionate line rather than a hairline.
    public var jointFraction: Double
    /// Per-unit tone spread, 0 (identical clones) … 1 (strong variation). Also the material's
    /// main defence against `TextureAudit.isFlatColor` when body and joint are close in value.
    public var toneVariation: Double
    /// Redraws the per-unit tone lottery.
    public var seed: UInt64

    public init(shape: UnitShape = .square,
                unitSizeMeters: Double = 0.1524,          // 6″ — the commonest tile there is
                bond: UnitBond = .stack,
                run: UnitRun = .horizontal,
                bodyColor: Vec3 = UnitPatternParams.defaultBodyColor,
                jointColor: Vec3 = UnitPatternParams.defaultJointColor,
                jointFraction: Double = 0.055,
                toneVariation: Double = 0.14,
                seed: UInt64 = 4) {
        self.shape = shape
        self.unitSizeMeters = Self.quantizeSize(unitSizeMeters)
        self.bond = bond
        self.run = run
        self.bodyColor = clampBand(bodyColor)
        self.jointColor = clampBand(jointColor)
        self.jointFraction = Swift.max(0.005, Swift.min(0.30, (jointFraction * 1000).rounded() / 1000))
        self.toneVariation = clamp01(toneVariation)
        self.seed = seed
    }

    /// Tolerant decode — a document written before `run` existed still loads.
    public init(from decoder: Decoder) throws {
        let d = try decoder.container(keyedBy: CodingKeys.self)
        let def = UnitPatternParams()
        self.init(shape: try d.decodeIfPresent(UnitShape.self, forKey: .shape) ?? def.shape,
                  unitSizeMeters: try d.decodeIfPresent(Double.self, forKey: .unitSizeMeters) ?? def.unitSizeMeters,
                  bond: try d.decodeIfPresent(UnitBond.self, forKey: .bond) ?? def.bond,
                  run: try d.decodeIfPresent(UnitRun.self, forKey: .run) ?? def.run,
                  bodyColor: try d.decodeIfPresent(Vec3.self, forKey: .bodyColor) ?? def.bodyColor,
                  jointColor: try d.decodeIfPresent(Vec3.self, forKey: .jointColor) ?? def.jointColor,
                  jointFraction: try d.decodeIfPresent(Double.self, forKey: .jointFraction) ?? def.jointFraction,
                  toneVariation: try d.decodeIfPresent(Double.self, forKey: .toneVariation) ?? def.toneVariation,
                  seed: try d.decodeIfPresent(UInt64.self, forKey: .seed) ?? def.seed)
    }

    /// The unit's short edge, DERIVED. Never stored — see `unitSizeMeters`.
    public var shortEdgeMeters: Double { unitSizeMeters / Double(shape.aspect) }

    /// The long edge in inches — what the number field shows in imperial documents.
    public var unitSizeInches: Double { unitSizeMeters / 0.0254 }

    // MARK: taste dials (single named levers — Danny to retune)

    /// SINGLE NAMED LEVER: the default tile face. A warm off-white glazed ceramic — the
    /// neutral that reads as "tile" before anyone picks a colour.
    public static let defaultBodyColor = Vec3(0.74, 0.72, 0.68)

    /// SINGLE NAMED LEVER: **white grout.** Not `1.0` — `TextureAudit.albedoOutOfRange`
    /// fails any channel above 0.94 and `isPlausible` requires ZERO such pixels, so "white"
    /// here is the brightest honest dielectric white. It still reads as white against the
    /// body colour above, which is what matters.
    public static let defaultJointColor = Vec3(0.90, 0.89, 0.87)

    /// **The widest joint worth offering**, as a fraction of the unit's long edge — 20% of a 6″
    /// tile is a 30 mm grout line, already past anything a tiler would set. Stated once here
    /// because the control's range has to agree with what the params accept; the `init` clamp
    /// stays wider (0.30) so a document or a preset can carry an unusual value without being
    /// silently rewritten. The BOTTOM of the offered range is not a constant — it is whatever
    /// the bake can draw, `TileLayout.minJointFraction`.
    public static let maxOfferedJointFraction = 0.20

    // MARK: bounded key space

    /// **Quantise the size on write.** A free number input plus colour pickers is an unbounded
    /// cache key, and customized atlas slices are never evicted (CLAUDE.md § Material baking
    /// rules) — so a user scrubbing a stepper would mint a permanent 512² slice per keystroke.
    /// 1 mm is far below anything visible at tile scale, and it makes the key space finite.
    static func quantizeSize(_ m: Double) -> Double {
        Swift.max(0.02, Swift.min(1.2, (m * 1000).rounded() / 1000))
    }
}

// MARK: - Layout: the arithmetic that makes SIZE a free number

/// **How a real-world unit size becomes a whole grid on a square bake.** Pure arithmetic, no
/// pixels — so the whole size contract is testable without baking anything.
///
/// The old spine had this backwards. It fixed the surface run at 2 m and SNAPPED the unit
/// count to a divisor of the 512 bake, which is why size had to be a *picker* over
/// `offeredSizesMeters` rather than a number. That divisor rule was folklore: measured
/// 2026-08-10, every integer count 2…32 scores ≤0.001 against a seam ceiling of 3.0,
/// non-divisors included (`testEveryIntegerTileCountIsSeamless`), and `unitPavers` had been
/// shipping 18 cells on a 512 bake the whole time. Only WHOLE counts are required.
///
/// So the relationship is inverted: the count follows the size, and **the run is derived from
/// the count** (`runMeters = cols × unitSize`). Any size is then realised exactly and stays
/// seamless — and because `cols` tracks `preferredRun / size`, the run stays near the
/// surface's natural 2 m instead of swinging with the size (which would have made a 5 cm
/// mosaic repeat every 40 cm — manufacturing the very "repeating tiling is a mess" complaint
/// this work exists to fix).
public struct TileLayout: Equatable, Hashable, Sendable {
    /// Units across the bake.
    public let cols: Int
    /// Units down the bake. Always `cols × aspect`, and always a multiple of the bond's
    /// row cycle so the offset pattern wraps.
    public let rows: Int
    /// World metres one bake spans — the material's `uvScale`.
    public let runMeters: Double
    /// The atlas resolution this layout was derived for. Stored because the JOINT floor below is
    /// a function of it, and the control that offers the joint width must quote the same number
    /// the baker will use.
    public let bakeSize: Int

    /// The unit long edge this layout actually produces. Equals the asked-for size exactly on
    /// an adjustable surface; quantised to `metresPerRepeat / cols` on a fixed one.
    public var realisedSizeMeters: Double { runMeters / Double(cols) }

    /// Texels across the unit's SHORT edge at a given bake resolution — the resolution budget
    /// that bounds how fine a grid is worth baking.
    public func shortEdgeTexels(bakeSize: Int) -> Int { bakeSize / Swift.max(1, rows) }

    // MARK: - The joint the bake can actually draw

    /// Texels a joint must span before it can be drawn at all. Below this the line aliases into
    /// a dotted mess and the wrap stops matching, so the baker widens it to this instead.
    public static let minJointTexels: Double = 1.6

    /// **The thinnest joint this layout can draw, as a `jointFraction`.** Any smaller ask bakes
    /// identical pixels — so this is the bottom of the range a width control may offer, and
    /// offering below it is a dead stretch of slider (it was: a default 6″ tile floors at 0.05,
    /// exactly where the default width sits, so dragging DOWN from the start did nothing).
    ///
    /// Identical on both axes by construction: the u floor is `2 × minJointTexels / (bake/cols)`
    /// and the v floor is the same number ÷ `aspect` measured in short-edge units — which is
    /// what makes the floored joint stay physically SQUARE, like the authored one.
    public var minJointFraction: Double {
        2 * Self.minJointTexels * Double(cols) / Double(Swift.max(bakeSize, 1))
    }

    /// The width the bake will ACTUALLY draw for an asked-for `jointFraction`.
    public func effectiveJointFraction(_ asked: Double) -> Double {
        Swift.max(asked, minJointFraction)
    }

    /// The realised joint width in METRES — the number to show the user, because it is the one
    /// they get. Quoted against `realisedSizeMeters`, so a quantised size can't make it a lie.
    public func jointWidthMeters(_ asked: Double) -> Double {
        effectiveJointFraction(asked) * realisedSizeMeters
    }

    /// **One texel of joint, as a `jointFraction` step.** The joint's drawn width is quantised to
    /// texels, so this is both the smallest change that moves a pixel and the step a control must
    /// use to keep its key space finite: every customised bake is cached by `params.hashValue` and
    /// never evicted, so a continuous slider would mint a permanent 512² atlas slice per drag tick
    /// while showing the user the same picture. Same reasoning as `quantizeSize`.
    public var jointFractionTexelStep: Double { Double(cols) / Double(Swift.max(bakeSize, 1)) }

    /// Snap an asked width onto the texel grid and into the range a CONTROL may offer. Only the
    /// control quantises — a document or preset carrying an unusual width is baked as authored
    /// (floored, not snapped), so opening a file can never silently rewrite it.
    public func quantizeJointFraction(_ asked: Double) -> Double {
        let step = jointFractionTexelStep
        let snapped = (asked / step).rounded() * step
        return Swift.min(Swift.max(minJointFraction, snapped),
                         Swift.max(minJointFraction, UnitPatternParams.maxOfferedJointFraction))
    }

    /// The range a width control should offer: floor → the widest joint worth setting. **Empty
    /// (`lowerBound == upperBound`) when the grid is so fine that the floor has reached the top**
    /// — at which point there is no width to choose and the honest UI is a sentence, not a slider.
    public var offeredJointFractions: ClosedRange<Double> {
        let lo = Swift.min(minJointFraction, UnitPatternParams.maxOfferedJointFraction)
        return lo...UnitPatternParams.maxOfferedJointFraction
    }

    // MARK: constraints

    /// Minimum units per bake. Below ~12 the per-unit tone lottery has too few draws and the
    /// repeat reads as visible clones rather than variation.
    public static let minUnitsPerBake = 12
    /// Minimum texels across a unit's short edge. Below 16 the joint drops under the
    /// 1.6-texel floor the baker enforces and the line comes out wider than authored.
    public static let minShortEdgeTexels = 16

    /// **The world patch one tile bake covers, when the material gets to choose it.**
    ///
    /// On an adjustable surface the run is ours entirely (it is derived from the grid), so this is
    /// a free choice — and it is the lever that decides how fine a GROUT LINE the atlas can draw.
    /// Measured: the joint is floored at 1.6 texels, so at a 512 bake the thinnest drawable joint
    /// is `3.2 × (run / 512)` — a near-constant *physical* width regardless of tile size.
    ///
    /// | run | mm per texel | thinnest joint |
    /// |---|---|---|
    /// | 2.0 m | 3.9 mm | **12.5 mm** — far fatter than real grout |
    /// | 1.2 m | 2.3 mm | **7.5 mm** |
    /// | 0.6 m | 1.2 mm | 3.8 mm, but only ~4 units per bake at 6″ |
    ///
    /// 1.2 m is the balance: it roughly halves the minimum joint while keeping ≥16 units per bake
    /// across the whole practical size range. Shortening the run costs little HERE specifically
    /// because tile is a **coherent** pattern — it opts out of the hex de-repeat precisely because
    /// real tile is supposed to repeat, so a tighter period reads as "tile", not as "a repeating
    /// texture". (Do not generalise this to a stochastic material, where a short period is exactly
    /// the artefact the de-repeat exists to hide.)
    ///
    /// TASTE — flagged for Danny: a genuine 3 mm grout line is below this atlas's resolution at any
    /// sane repeat. Lower this to buy finer joints at the cost of a shorter repeat.
    public static let targetRunMeters: Double = 1.2

    /// Derive the grid for `params` on a surface. `tiling.isAdjustable` says whether the mesh's
    /// UV period can be set from the material (the structural wall/floor/ceiling convert path)
    /// or is fixed by a mesh baked earlier (every instanced placeable).
    public static func of(_ params: UnitPatternParams, on tiling: SurfaceTiling,
                          bakeSize: Int = MaterialGenerator.bakeSize) -> TileLayout {
        let aspect = params.shape.aspect
        let cycle = params.bond.rowCycle

        // Resolution ceiling and variation floor, DERIVED from the two constraints above
        // rather than hand-tabled per shape.
        let maxCols = Swift.max(1, (bakeSize / minShortEdgeTexels) / aspect)
        let minCols = Swift.max(1, Int((Double(minUnitsPerBake) / Double(aspect)).squareRoot().rounded(.up)))

        // On an adjustable surface the run is DERIVED from the grid, so we are free to pick the
        // grid that draws the crispest joint (`targetRunMeters`) rather than the surface's nominal
        // patch size. On a fixed-period mesh we get no such choice — the count must match the
        // period the mesh already baked, or the tile size is a lie.
        let target = tiling.isAdjustable ? Swift.min(tiling.metresPerRepeat, targetRunMeters)
                                         : tiling.metresPerRepeat
        let wanted = target / Swift.max(params.unitSizeMeters, 1e-4)
        var cols = Swift.max(Swift.min(minCols, maxCols), Swift.min(maxCols, Int(wanted.rounded())))

        // `rows = cols × aspect` must be a multiple of the bond cycle or the offset pattern
        // cannot wrap at the v-seam. Step UP to the next satisfying count, and only step down
        // if that would breach the resolution ceiling.
        func rowsOK(_ c: Int) -> Bool { (c * aspect) % cycle == 0 }
        if !rowsOK(cols) {
            var up = cols
            while up <= maxCols && !rowsOK(up) { up += 1 }
            if up <= maxCols {
                cols = up
            } else {
                var down = cols
                while down > 1 && !rowsOK(down) { down -= 1 }
                cols = Swift.max(1, down)
            }
        }

        let rows = Swift.max(1, cols * aspect)
        // THE INVERSION: on an adjustable surface the run is derived from the grid, so the
        // asked-for size is realised EXACTLY. On a fixed-period mesh the run is not ours to
        // set, so the size quantises to `metresPerRepeat / cols` instead — the control stays
        // live everywhere, and is honest about where it cannot be exact.
        let run = tiling.isAdjustable ? Double(cols) * params.unitSizeMeters
                                      : tiling.metresPerRepeat
        return TileLayout(cols: cols, rows: rows, runMeters: run, bakeSize: bakeSize)
    }
}

// MARK: - The one tile generator

public extension MaterialGenerator {

    /// **Tile — one generator for every shape and bond.** Replaces the near-duplicate pair
    /// `ceramicTile` (square, stack) + `wallTile` (3:1-or-2:1-depending-on-the-path, running).
    /// Those two disagreed with each other in a way nobody could see: `wallTile`'s default was
    /// `cols:4, rows:12` (3:1) while its params overload used `rows: cols * 2` (2:1), so
    /// *customizing a wall tile silently changed its shape*. One generator, one layout, one
    /// proportion.
    ///
    /// Structure is unchanged from the originals where they were right: a regular grid kept on
    /// classic UV tiling (NOT stochastic — tile is a coherent pattern and opts out of the hex
    /// de-repeat), per-unit hash tone so the repeat isn't identical clones, and the joint as an
    /// **independent recessed layer** with its own colour/roughness/depth over the body.
    ///
    /// Texel-centre UVs throughout — the sibling generators (`wallTile`, `unitPavers`) already
    /// used them and documented why: with `u = x/size` the last column lands mid-body while
    /// column 0 lands in the joint, a half-texel wrap asymmetry. `ceramicTile` alone did not,
    /// and the divisor snap had been masking it.
    /// `faceImage` supplies the decoded pixels for `TileParams.face == .image(...)`. `DaydreamCore`
    /// holds no image data, so the render layer resolves the document id and passes the channels
    /// in; nil (an id that no longer exists, or a plain procedural tile) falls back to the
    /// generated face rather than failing to bake.
    static func tile(size: Int = MaterialGenerator.bakeSize,
                     params: TileParams = TileParams(),
                     tiling: SurfaceTiling = .librarySwatch,
                     faceImage: MaterialChannels? = nil) -> MaterialChannels {
        let p = params.pattern
        let layout = TileLayout.of(p, on: tiling, bakeSize: size)
        let cols = layout.cols, rows = layout.rows
        var ch = MaterialChannels(size: size, category: .tile)
        // S2.5 half 2 — per-physical-tile de-repeat metadata, from the SAME layout that
        // draws the grid. A bonded u-axis ships 0 (courses vary as wholes) so the engine's
        // cell lattice can never cut through an offset tile mid-body; a vertical run is a
        // quarter-turn of the lattice, so the axes (and the bonded axis) swap with it.
        let bonded = p.bond != .stack
        ch.patternCells = p.run == .vertical
            ? Vec2(Double(rows), bonded ? 0 : Double(cols))
            : Vec2(bonded ? 0 : Double(cols), Double(rows))
        ch.patternJitter = 0.04

        // Joint half-widths per axis. `jointFraction` is a fraction of the LONG edge; the v
        // axis is measured in short-edge units, so its half-width scales by the aspect — that
        // is what makes the joint physically SQUARE on a rectangular tile instead of a
        // letterboxed grid (the bug `wallTile` papered over with two hand-picked constants).
        //
        // FLOORED AT `TileLayout.minJointTexels` on each axis. On a fine grid a proportional joint
        // falls below one texel, the line aliases into a dotted mess and the wrap stops matching.
        // A joint that cannot be drawn is not a joint, so it widens to the thinnest line the bake
        // can hold — which is also what fine mosaic genuinely looks like.
        //
        // The floor is taken from `layout`, NOT re-derived here: the width CONTROL has to offer
        // exactly the range this baker will honour, and two copies of that arithmetic is how the
        // slider ended up with a dead bottom quarter nobody could see.
        let aspect = Double(p.shape.aspect)
        let texelsU = Double(size) / Double(Swift.max(1, cols))
        let texelsV = Double(size) / Double(Swift.max(1, rows))
        let jf = layout.effectiveJointFraction(p.jointFraction)
        let halfU = jf / 2
        let halfV = jf * aspect / 2
        // Antialias band, in tile units, ≈1 texel wide on each axis.
        let softU = Swift.max(1.0 / Swift.max(texelsU, 1), 0.004)
        let softV = Swift.max(1.0 / Swift.max(texelsV, 1), 0.004)

        let bevel = params.edge.bevelFraction
        let bodyRough = params.finish.bodyRoughness
        // An imported face is only used when the params actually ask for one — so a stale
        // `faceImage` can't leak onto a tile the user switched back to procedural.
        let face = params.face.imageDocumentID == nil ? nil : faceImage

        for y in 0..<size {
            for x in 0..<size {
                // A VERTICAL run is an exact quarter turn of the lattice: swap the sampling axes
                // and the grid counts. On a square bake that maps the pattern onto itself, so the
                // vertical layout is seamless by construction rather than by luck.
                let su = (Double(x) + 0.5) / Double(size)
                let sv = (Double(y) + 0.5) / Double(size)
                let u = p.run == .vertical ? sv : su
                let v = p.run == .vertical ? su : sv

                let tv = v * Double(rows)
                let tj = Int(floor(tv))
                let tu = u * Double(cols) + p.bond.offset(row: tj)
                let ti = Int(floor(tu))
                let fx = tu - floor(tu), fy = tv - floor(tv)

                // Wrapped unit indices: the per-unit tone must be identical either side of the
                // u/v seam, so it is hashed on the WRAPPED index, not the raw one.
                let tiW = ((ti % cols) + cols) % cols
                let tjW = ((tj % rows) + rows) % rows

                // Distance to the nearest joint on each axis, in that axis's tile units.
                let dU = Swift.min(fx, 1 - fx), dV = Swift.min(fy, 1 - fy)
                let gU = 1 - smoothstep(halfU, halfU + softU, dU)
                let gV = 1 - smoothstep(halfV, halfV + softV, dV)
                let joint = Swift.max(gU, gV)

                // Per-unit discrete tone — the thing that stops a grid reading as clones.
                let th = Noise.hash2(tiW, tjW, p.seed)
                let tone = 1 + (Noise.unit(th) - 0.5) * p.toneVariation
                let warp = Noise.fbmTiled(u, v, baseCells: Swift.max(cols, 2) * 3, octaves: 3,
                                          seed: p.seed ^ 0x11)
                let macro = 0.97 + 0.06 * Noise.fbmTiled(u, v, baseCells: 2, octaves: 3,
                                                         seed: p.seed ^ 0x22)

                // Glaze pooling: fired glaze runs thicker (darker, glossier) toward the unit
                // edge and thins at the centre. Measured from the normalised edge distance so
                // it is symmetric per unit and stays seamless on both axes.
                let edgeN = Swift.min(dU / Swift.max(halfU, 1e-6), dV / Swift.max(halfV, 1e-6))
                let pool = 1 - smoothstep(1.0, 3.0, edgeN)

                // THE UNIT FACE. Procedural: the body colour, per-unit tone and glaze pooling.
                // Imported: the photo, sampled by the WITHIN-UNIT coordinate so one copy lands on
                // each tile — which is what lets grout, size and bond keep working over it. The
                // per-unit tone still applies (real tile from one box does vary), the glaze pooling
                // does not (the photo carries its own shading already).
                var bodyCol: Vec3
                if let face {
                    let fu = Swift.min(Swift.max(fx, 0), 0.999999)
                    let fv = Swift.min(Swift.max(fy, 0), 0.999999)
                    let sx = Int(fu * Double(face.size)), sy = Int(fv * Double(face.size))
                    bodyCol = face.albedo[face.idx(sx, sy)] * tone
                } else {
                    bodyCol = p.bodyColor * tone * (0.96 + 0.08 * warp) * (1 - 0.09 * pool)
                }
                ch.albedo[ch.idx(x, y)] = clampBand(mix(bodyCol, p.jointColor, joint) * macro)

                let micro = (Noise.fbmTiled(u, v, baseCells: 30, octaves: 2,
                                            seed: p.seed ^ 0x33) - 0.5) * 0.04
                ch.roughness[ch.idx(x, y)] =
                    clamp01(mix(bodyRough, 0.85, joint) + micro - 0.05 * pool)

                // Height: the joint is recessed. A BEVELLED edge additionally chamfers the
                // unit's own rim — a ramp over the band just inside the joint — which is free
                // here because the normal is derived from this field anyway.
                let chamfer = bevel > 0 ? (1 - smoothstep(0, bevel, Swift.min(dU, dV * aspect))) : 0
                ch.height[ch.idx(x, y)] = clamp01(1 - joint * 0.7 - chamfer * bevel * 6)
            }
        }

        ch.clearcoat = params.finish.clearcoat
        ch.deriveNormals(strength: 6)
        // Glaze orange-peel: full strength in the NORMAL, a quarter of it in the occlusion —
        // a levelled vitreous film tilts light but traps none. See `glazeDetailOcclusionShare`.
        addMicroDetail(&ch, seed: p.seed ^ 0xF2, baseCells: 80, strength: 0.4,
                       occlusionStrength: 0.4 * MaterialGenerator.glazeDetailOcclusionShare)
        return ch
    }
}

// MARK: - TileParams

/// A tile's full customization: the shared unit pattern plus the two things that are genuinely
/// tile-only — how the unit's rim is finished, and how glossy the glaze is.
public struct TileParams: Equatable, Hashable, Sendable, Codable {
    public var pattern: UnitPatternParams
    public var finish: SurfaceFinish
    public var edge: TileEdge
    public var face: TileFace

    public init(pattern: UnitPatternParams = UnitPatternParams(),
                finish: SurfaceFinish = .satin,
                edge: TileEdge = .square,
                face: TileFace = .procedural) {
        self.pattern = pattern; self.finish = finish; self.edge = edge; self.face = face
    }

    /// Tolerant decode — a document written before `face` existed still loads.
    public init(from decoder: Decoder) throws {
        let d = try decoder.container(keyedBy: CodingKeys.self)
        let def = TileParams()
        self.init(pattern: try d.decodeIfPresent(UnitPatternParams.self, forKey: .pattern) ?? def.pattern,
                  finish: try d.decodeIfPresent(SurfaceFinish.self, forKey: .finish) ?? def.finish,
                  edge: try d.decodeIfPresent(TileEdge.self, forKey: .edge) ?? def.edge,
                  face: try d.decodeIfPresent(TileFace.self, forKey: .face) ?? def.face)
    }

    /// Convenience for the common "just give me a size" construction (and every existing
    /// call site). Everything else takes its default.
    public init(tileSizeMeters: Double, groutFraction: Double = 0.055) {
        self.init(pattern: UnitPatternParams(unitSizeMeters: tileSizeMeters,
                                             jointFraction: groutFraction))
    }

    // Tile-facing names for the shared spine's fields, so call sites read as tile code.
    public var tileSizeMeters: Double {
        get { pattern.unitSizeMeters }
        set { pattern.unitSizeMeters = UnitPatternParams.quantizeSize(newValue) }
    }
    public var tileSizeInches: Double { pattern.unitSizeInches }
    public var groutFraction: Double {
        get { pattern.jointFraction }
        set { pattern.jointFraction = Swift.max(0.005, Swift.min(0.30, newValue)) }
    }
    public var groutColor: Vec3 {
        get { pattern.jointColor }
        set { pattern.jointColor = clampBand(newValue) }
    }
    public var bodyColor: Vec3 {
        get { pattern.bodyColor }
        set { pattern.bodyColor = clampBand(newValue) }
    }
    public var shape: UnitShape {
        get { pattern.shape }
        set { pattern.shape = newValue }
    }
    public var bond: UnitBond {
        get { pattern.bond }
        set { pattern.bond = newValue }
    }

    /// The layout this tile produces on a surface — what the UI reads to show the user the
    /// size they will actually get.
    public func layout(on tiling: SurfaceTiling) -> TileLayout { TileLayout.of(pattern, on: tiling) }

    /// What the retired `wall-tile` id bakes as. A glossy white running-bond subway at the real
    /// 2:1 proportion — which is what its *customized* path always produced (`rows: cols * 2`),
    /// even though its uncustomized default was 3:1. Documents that persisted `wall-tile` keep
    /// rendering a subway tile; they just can't be *picked* as a second material any more.
    public static let legacyWallTile = TileParams(
        pattern: .init(shape: .rectangle, unitSizeMeters: 0.1524, bond: .running,
                       bodyColor: Vec3(0.86, 0.86, 0.84), jointFraction: 0.045),
        finish: .gloss)

    // MARK: named presets

    /// The curated looks, as `(name, params)`. These REPLACE the old per-size variant list
    /// (which existed only because size had to be a picker) — a variant is now a genuinely
    /// different tile, not the same tile at another scale.
    public static let presets: [(name: String, params: TileParams)] = [
        ("Subway 3×6", TileParams(pattern: .init(shape: .rectangle, unitSizeMeters: 0.1524,
                                                 bond: .running), finish: .gloss)),
        ("6″ Square", TileParams(pattern: .init(shape: .square, unitSizeMeters: 0.1524))),
        ("12″ Square", TileParams(pattern: .init(shape: .square, unitSizeMeters: 0.3048))),
        ("Penny Mosaic", TileParams(pattern: .init(shape: .square, unitSizeMeters: 0.05,
                                                   jointFraction: 0.11), finish: .gloss)),
        ("12×24 Plank", TileParams(pattern: .init(shape: .rectangle, unitSizeMeters: 0.6096,
                                                  bond: .thirdRunning), finish: .matte)),
        ("Slate Floor", TileParams(pattern: .init(shape: .square, unitSizeMeters: 0.3048,
                                                  bodyColor: Vec3(0.26, 0.27, 0.28),
                                                  jointColor: Vec3(0.34, 0.34, 0.33),
                                                  toneVariation: 0.22), finish: .matte)),
    ]
}

/// **What the tile's FACE is made of.**
///
/// Danny imports photographs of tile. Treated as a plain repeating surface, such a photo fights
/// every control the tile material has: the image already contains grout, so setting a grout colour
/// does nothing, and setting a tile size lays a grid over a picture that has its own grid — the
/// double-grid that makes an imported tile photo look wrong however you tune it.
///
/// Pointing the tile's face AT the image instead inverts that. The photo becomes the surface of one
/// unit, and shape, size, bond, grout colour and grout width all keep working, because they are
/// drawn procedurally *around* it. (Port of the same idea from `~/Sites/roominate`'s
/// `generateTileFromImage`.)
public enum TileFace: Equatable, Hashable, Sendable, Codable {
    /// The generated glazed-ceramic face — colour, tone variation and glaze pooling.
    case procedural
    /// One unit's face is an imported image, referenced by its `DocumentMaterial` id. The id is
    /// resolved by the render layer (`DaydreamCore` holds no image data), which hands the decoded
    /// channels to the generator; an id that no longer exists falls back to `.procedural` rather
    /// than failing to bake.
    case image(documentID: String)

    public var imageDocumentID: String? {
        if case .image(let id) = self { return id }
        return nil
    }
}

/// How a tile's rim is finished. A bevel is a real chamfer on the height field, so it catches
/// a highlight along every joint — the look of a classic bevelled subway tile.
public enum TileEdge: String, Equatable, Hashable, Sendable, Codable, CaseIterable {
    case square, bevelled

    /// Chamfer width as a fraction of the unit, 0 for a square rim.
    public var bevelFraction: Double { self == .bevelled ? 0.06 : 0 }

    public var displayName: String { self == .square ? "Square edge" : "Bevelled" }
}
