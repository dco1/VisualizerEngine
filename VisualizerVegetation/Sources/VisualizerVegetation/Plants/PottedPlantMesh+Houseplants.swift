import Foundation
import simd
import VisualizerMaterials

extension LeafSilhouette {
    /// A fiddle-leaf fig blade as the plant grows it: a U-rounded base (not the tapered club the old
    /// outline had — cupped, that read as a calla-lily trumpet), a small lower bout, a waist a little
    /// under halfway, then the broad rounded shoulders two-thirds up and a blunt, nearly truncate tip.
    /// Measured on a 2D port of `smooth` before it was ever rendered: worst turn 21° at
    /// `subdivisions: 3` (a crease starts reading at ~40°); widest at 0.65 of the length; waist
    /// 0.49 of the widest.
    public static let fiddleLeafFigBlade = LeafSilhouette(
        name: "fiddleLeafFigBlade",
        hero: [(0.00, 0.00), (0.02, 0.18), (0.06, 0.34), (0.13, 0.44), (0.22, 0.47), (0.34, 0.42),
               (0.46, 0.47), (0.60, 0.66), (0.74, 0.82), (0.86, 0.84), (0.94, 0.68), (0.985, 0.36),
               (1.00, 0.00)])

    /// The smoothed half-margin sampled at chosen `v` rows — a row where the outline TURNS (the
    /// rounded base, the blunt tip) and fewer where it runs straight. 17 such rows round the fig's
    /// blade as well as `subdivisions: 2`'s 25 do (worst turn 28.7° against 25.5°, both under the
    /// ~40° a crease starts at) for two-thirds of the triangles. The ends are pinned to `u = 0`.
    public func margin(atRows rows: [Double]) -> [Control] {
        let dense = margin(tier: .hero, subdivisions: 8)
        return rows.map { v in Control(v, v <= 0 || v >= 1 ? 0 : GardenPlantMesh.halfWidth(dense, atV: v)) }
    }

    /// A snake-plant sword: clasped narrow at the soil, widest a third of the way up, then a long even
    /// taper to a hard point — stiff, not strap-like.
    public static let snakePlantSword = LeafSilhouette(
        name: "snakePlantSword",
        hero: [(0.00, 0.00), (0.03, 0.62), (0.12, 0.86), (0.30, 1.00), (0.52, 0.94), (0.72, 0.74),
               (0.87, 0.46), (0.96, 0.18), (1.00, 0.00)])
}

/// The five FOLIAGE houseplants, each grown by its own habit and painted leaf by leaf.
///
/// What this replaced: one arranger (`FoliagePlan` over a golden spiral) placing one kind of card —
/// a strip blade stamped one flat green by the render bridge — tuned per style by a table of
/// numbers. It could not give a fig a rounded leaf base, a monstera a single split, a snake plant
/// any thickness or banding, a fern more than a handful of wiry fronds, or any plant two leaves of
/// different colours, and those are exactly the tells that read as a video-game asset beside the
/// wildflowers (whose generator already grows each kind by its habit). So each style here is its
/// own function, the way `GardenPlantMesh`'s species are:
///
///  - **Fiddle-leaf fig** — a bare woody cane; 15 leathery violin blades on short stalks up its top
///    two-thirds, older leaves larger and flatter, this season's flush small, upright and lime.
///    `LeafSheet` blades: a round cup, a sunken pale midrib, a herringbone of lateral veins with
///    the blade quilted between them, a wavy margin, a paler underside.
///  - **Monstera** — leaves on long arching petioles from a stub at the soil; heart-shaped blades
///    cut by splits and pierced by holes (`LeafSheet.emitMonsteraHalf`), each half its own cuts;
///    the newest leaf smaller, uncut, lime.
///  - **Snake plant** — clumps of stiff, THICK swords (`FleshyLeaf` solids, the aloe's kit), dark
///    green crossed by grey-green bands.
///  - **Fern** — a fountain of twenty-two arching fronds, each a rachis carrying opposite pairs of
///    small keeled pinnae, lanceolate overall; the young fronds in the centre upright and paler.
///  - **Succulent** — an echeveria-style rosette of plump spoon leaves (`FleshyLeaf`), a farina-
///    matte sage, the tips blushing.
///
/// Everything is one `PaintedMesh`: one draw instance for the whole plant however many leaves
/// (the performance rule every placeable answers to), cached by the bridge on the plant's params.
extension PottedPlantMesh {

    /// A foliage style's painted leaves, plus anything woody that belongs with the trunk.
    public struct Foliage: Sendable {
        public var leaves: PaintedMesh
        public var woody: Mesh3
        public init(leaves: PaintedMesh = PaintedMesh(), woody: Mesh3 = Mesh3()) {
            self.leaves = leaves; self.woody = woody
        }
    }

    /// Grow a foliage style's leaves (nil for the bouquet and tree styles, which build their own).
    public static func houseplantFoliage(params: some PottedPlantGeometry, soilY: Double,
                                         rng: inout SplitMix) -> Foliage? {
        switch params.plantStyle {
        case .fiddleLeafFig: return figFoliage(params: params, soilY: soilY, rng: &rng)
        case .monstera:      return monsteraFoliage(params: params, soilY: soilY, rng: &rng)
        case .snakePlant:    return snakePlantFoliage(params: params, soilY: soilY, rng: &rng)
        case .fern:          return fernFoliage(params: params, soilY: soilY, rng: &rng)
        case .succulent:     return succulentFoliage(params: params, soilY: soilY, rng: &rng)
        case .flowers, .driedSpray, .christmasTree: return nil
        }
    }

    /// How many leaves (fronds, swords…) a style carries at `density` 1, and how that scales.
    static func leafCount(_ base: Double, _ density: Float, minimum: Int = 3) -> Int {
        max(minimum, Int((base * Double(density)).rounded()))
    }

    /// A leaf's frame. The midrib grows out along `azimuth`, pitched `tilt` off vertical (0 straight
    /// up, π/2 flat out); `roll` turns the blade about its own midrib. `n` is the UPPER (adaxial)
    /// face: toward the axis the leaf grows from and up — the face a leaf turns to the light.
    public static func leafFrame(azimuth: Double, tilt: Double, roll: Double = 0) -> (x: Vec3, y: Vec3, n: Vec3) {
        let out = Vec3(cos(azimuth), 0, sin(azimuth))
        let y = normalize3(out * sin(tilt) + Vec3(0, cos(tilt), 0))
        let x0 = Vec3(-sin(azimuth), 0, cos(azimuth))
        let n0 = normalize3(cross3(x0, y))
        let x = normalize3(x0 * cos(roll) + n0 * sin(roll))
        let n = normalize3(n0 * cos(roll) - x0 * sin(roll))
        return (x, y, n)
    }

    /// A painted stalk (petiole, rachis): a smooth tube swept along `path`, tapering `rBase` → `rTip`.
    static func paintedStalk(_ path: [Vec3], rBase: Double, rTip: Double, sides: Int, color: Vec3) -> PaintedMesh {
        guard path.count >= 2 else { return PaintedMesh() }
        var m = Mesh3()
        let n = path.count
        let scales = (0 ..< n).map { 1 + (rTip / rBase - 1) * Double($0) / Double(n - 1) }
        guard m.sweep(profile: .circle(radius: rBase, segments: sides), along: path, scales: scales) else {
            return PaintedMesh()
        }
        return PaintedMesh(mesh: m.smoothed(), color: color)
    }

    // MARK: - Fiddle-leaf fig

    static func figFoliage(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix) -> Foliage {
        let style = PlantStyle.fiddleLeafFig
        let palette = style.foliagePalette
        let size = params.plantSize
        let stemTop = soilY + max(0.02, size * stemFraction(style))
        let count = leafCount(15, params.foliageDensity, minimum: 4)
        // Rows close at the rounded base and tip (where the OUTLINE turns) and split through the long
        // middle (where the blade's DROOP curves it): a row there that spans a tenth of the leaf is a
        // flat facet the live canvas's screen-space AO reads as a crease — one soft dark band per
        // row across a drooping leaf's underside (measured: the bands halve in pitch and fade when
        // the rows double, and vanish from the ray-traced still). Splitting the spans ≥ 6 % of the
        // blade takes 17 rows to 27.
        let margin = LeafSilhouette.fiddleLeafFigBlade.margin(atRows: [
            0, 0.015, 0.04, 0.08, 0.11, 0.14, 0.18, 0.22, 0.265, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61,
            0.655, 0.70, 0.74, 0.78, 0.815, 0.85, 0.88, 0.91, 0.955, 0.985, 1])
        let aspect = 0.76
        let ribHalf = 0.014
        // No painted lateral veins: a vein is a thin diagonal line and this grid samples it at
        // one interior column per half-blade — it would paint a row of SPOTS, not a vein. The
        // midrib gets its own fixed-width column (`midribBand`), so it is crisp; the herringbone
        // needs a vein-aligned grid (the monstera's) or a leaf texture, not more of this one.
        let veins: LeafSheet.LateralVeins? = nil
        var out = Foliage()
        let start = rng.unit() * 2 * Double.pi
        for i in 0 ..< count {
            let t = count > 1 ? Double(i) / Double(count - 1) : 1   // 0 the oldest, lowest → 1 the newest
            let along = 0.34 + 0.66 * pow(t, 0.8)                     // the crown fills the top two-thirds
            let stationY = soilY + (stemTop - soilY) * along
            let azimuth = start + Double(i) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.35
            let youth = smoothstep(0.66, 1.0, t)                       // the top few: this season's flush
            // Older leaves are larger and held flatter; the flush is small and upright.
            let length = size * (0.31 - 0.10 * t) * (0.86 + rng.unit() * 0.28)
            let tilt = 1.05 - 0.55 * t + (rng.unit() - 0.5) * 0.28
            let roll = (rng.unit() - 0.5) * 0.45
            let tone = FoliagePaint.tone(youth: youth, palette: palette, rng: &rng)
            let outDir = Vec3(cos(azimuth), 0, sin(azimuth))

            // A short, stout petiole straight off the cane (Danny, 2026-09-12: a fig leaf branches
            // off the upward trunk — no out-and-down hook) onto the blade's base. Painted the stalk's
            // own olive-brown, never the blade green (the "stray wire" read of 2026-09-11).
            let trunkR = stemRadius(style, at: along)
            let petiole = length * (0.09 + 0.05 * rng.unit())
            let attach = Vec3(0, stationY, 0) + outDir * (trunkR * 0.7)
            let pDir = normalize3(outDir * 0.75 + Vec3(0, 0.66, 0))
            let bladeBase = attach + pDir * petiole
            let stalkR = min(trunkR * 0.5, max(0.0025, length * 0.018))
            out.leaves.append(paintedStalk([attach, attach + pDir * (petiole * 0.5), bladeBase],
                                           rBase: stalkR, rTip: stalkR * 0.75, sides: 6, color: palette.stalk))

            let f = leafFrame(azimuth: azimuth, tilt: tilt, roll: roll)
            var s = LeafSheet.Surface(base: bladeBase, xAxis: f.x, yAxis: f.y, normal: f.n,
                                      length: length, halfWidth: aspect / 2)
            s.cup = -0.01 + rng.unit() * 0.06
            s.droop = 0.06 + 0.10 * (1 - t) + rng.unit() * 0.04
            s.midribGroove = 0.005
            s.midribHalfWidth = ribHalf * 1.6
            s.ripple = 0.006 + rng.unit() * 0.006
            s.rippleWaves = 2.3 + rng.unit() * 1.2
            s.ripplePhase = rng.unit() * 2 * Double.pi
            s.twist = (rng.unit() - 0.5) * 0.30
            var leaf = PaintedMesh()
            LeafSheet.emitBlade(into: &leaf, surface: s, margin: margin, aspect: aspect,
                                columns: [0.55], midribBand: ribHalf, veins: veins) { site, face in
                FoliagePaint.sheet(palette, tone, site: site, face: face, ribHalfWidth: ribHalf)
            }
            out.leaves.append(leaf.smoothed())
        }
        return out
    }

    // MARK: - Monstera

    /// One monstera blade's cuts for one half, seeded. A juvenile (`maturity` near 0) leaf is entire;
    /// a mature one carries 5–7 splits of varied depth, some with a fenestration in line inside them.
    public static func monsteraCuts(maturity: Double, rng: inout SplitMix) -> LeafSheet.MonsteraCuts {
        let n = maturity < 0.25 ? 0 : Int((4 + 3 * maturity).rounded())
        var splits: [LeafSheet.MonsteraCuts.Split] = []
        var holes: [LeafSheet.MonsteraCuts.Hole] = []
        guard n > 0 else { return .init(splits: [], holes: []) }
        let a0 = 0.23, a1 = 0.86
        for k in 0 ..< n {
            let a = a0 + (a1 - a0) * Double(k) / Double(max(1, n - 1)) + (rng.unit() - 0.5) * 0.02
            let w = 0.016 + rng.unit() * 0.018
            let r = rng.unit()
            let depth = k == n - 1 ? 0.72 : (r < 0.20 ? 0.30 : (r < 0.80 ? 0.48 : 0.72))
            splits.append(.init(a: a, width: w, depth: depth))
            if depth >= 0.48, k < n - 1, rng.unit() < 0.55 * maturity {
                // In line with its split and exactly as wide — shared column edges, no sliver.
                holes.append(.init(a: a, width: w, r0: 0.18, r1: 0.30))
            }
        }
        return .init(splits: splits, holes: holes)
    }

    static func monsteraFoliage(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix) -> Foliage {
        let palette = PlantStyle.monstera.foliagePalette
        let size = params.plantSize
        let count = leafCount(7, params.foliageDensity, minimum: 3)
        var out = Foliage()
        let start = rng.unit() * 2 * Double.pi
        for i in 0 ..< count {
            let t = count > 1 ? Double(i) / Double(count - 1) : 1   // 0 oldest, outermost → 1 newest
            let azimuth = start + Double(i) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.4
            let outDir = Vec3(cos(azimuth), 0, sin(azimuth))
            let newest = i == count - 1 && count >= 3
            let maturity = newest ? 0.1 : 1 - 0.55 * t
            let youth = newest ? 1.0 : smoothstep(0.75, 1.0, t) * 0.5
            let tone = FoliagePaint.tone(youth: youth, palette: palette, rng: &rng)
            // Blade and where its petiole holds it: the old leaves low and far out, the new ones high
            // and close — a monstera spreads wider than it is tall.
            let blade = size * (newest ? 0.30 : 0.50 - 0.12 * t) * (0.90 + rng.unit() * 0.20)
            let reach = size * (0.30 - 0.16 * t) * (0.85 + rng.unit() * 0.3)
            let height = size * (0.34 + 0.34 * t) * (0.90 + rng.unit() * 0.20)
            let foot = Vec3(0, soilY + size * (0.02 + 0.06 * t), 0) + outDir * (size * 0.012)
            let bladeBase = Vec3(0, soilY + height, 0) + outDir * reach
            // The petiole: a thick green stalk arching out and up to the blade, bowed outward.
            let chord = bladeBase - foot
            let bow = outDir * (len3(chord) * 0.16)
            let path = (0 ... 6).map { k -> Vec3 in
                let u = Double(k) / 6
                return foot + chord * u + bow * (4 * u * (1 - u))
            }
            let petR = max(0.004, size * 0.010) * (newest ? 0.75 : 1)
            out.leaves.append(paintedStalk(path, rBase: petR, rTip: petR * 0.7, sides: 7,
                                           color: FoliagePaint.base(palette, LeafTone(youth: 0.3, value: tone.value), face: .upper) * 1.25))

            // The blade faces up and out, nearly flat, its tip hanging.
            let tilt = (newest ? 0.75 : 1.20 + 0.25 * (1 - t)) + (rng.unit() - 0.5) * 0.25
            let f = leafFrame(azimuth: azimuth + (rng.unit() - 0.5) * 0.3, tilt: tilt,
                              roll: (rng.unit() - 0.5) * 0.4)
            var s = LeafSheet.Surface(base: bladeBase, xAxis: f.x, yAxis: f.y, normal: f.n,
                                      length: blade, halfWidth: 0.48)
            s.cup = newest ? 0.10 : -0.03 - rng.unit() * 0.04   // a mature blade domes; the new one cups
            s.cupPower = 1.9
            s.droop = newest ? 0.04 : 0.10 + rng.unit() * 0.08
            s.midribGroove = 0.006
            s.midribHalfWidth = 0.02
            s.ripple = 0.006
            s.rippleWaves = 1.6
            s.ripplePhase = rng.unit() * 2 * Double.pi
            s.twist = (rng.unit() - 0.5) * 0.2
            let ribHalf = 0.013
            var leaf = PaintedMesh()
            for side in [1.0, -1.0] {
                let cuts = monsteraCuts(maturity: maturity, rng: &rng)
                LeafSheet.emitMonsteraHalf(into: &leaf, surface: s, cuts: cuts, side: side,
                                           spacing: 0.068) { site, face in
                    FoliagePaint.sheet(palette, tone, site: site, face: face, ribHalfWidth: ribHalf)
                }
            }
            out.leaves.append(leaf.smoothed())
        }
        return out
    }

    // MARK: - Snake plant

    /// The sword resampled to 33 EVENLY spaced rows. Its bands are a pattern along the blade, and
    /// the authored outline's rows bunch at the clasping base (where its shape changes fastest),
    /// leaving the middle — where the bands are seen — too sparse to carry them.
    static let snakePlantSwordRows: LeafSilhouette = {
        let dense = LeafSilhouette.snakePlantSword.margin(tier: .hero, subdivisions: 8)
        let n = 32
        return LeafSilhouette(name: "snakePlantSwordRows", hero: (0 ... n).map { k -> (Double, Double) in
            let v = Double(k) / Double(n)
            return (v, k == 0 || k == n ? 0 : GardenPlantMesh.halfWidth(dense, atV: v))
        })
    }()

    static func snakePlantFoliage(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix) -> Foliage {
        let palette = PlantStyle.snakePlant.foliagePalette
        let size = params.plantSize
        let clumps = max(2, min(5, Int((3.0 * Double(params.foliageDensity)).rounded())))
        var out = Foliage()
        let start = rng.unit() * 2 * Double.pi
        let up = Vec3(0, 1, 0)
        for c in 0 ..< clumps {
            // A rhizome clump: a fan of 2–4 swords in one plane, set off-centre in the pot.
            let caz = start + Double(c) * 2 * Double.pi / Double(clumps) + (rng.unit() - 0.5) * 0.8
            let cr = params.potRadius * (0.18 + 0.35 * rng.unit())
            let centre = Vec3(cos(caz) * cr, soilY - 0.015, sin(caz) * cr)
            let fanAz = rng.unit() * Double.pi
            let fanDir = Vec3(cos(fanAz), 0, sin(fanAz))
            let perClump = 2 + Int(rng.next() % 3)
            for k in 0 ..< perClump {
                let spread = perClump > 1 ? Double(k) / Double(perClump - 1) - 0.5 : 0
                let lean = spread * 0.40 + (rng.unit() - 0.5) * 0.10 + 0.06
                let outward = normalize3(Vec3(centre.x, 0, centre.z) + fanDir * 1e-3)
                let yAxis = normalize3(up + fanDir * sin(lean) + outward * (0.06 + 0.08 * rng.unit()))
                // The flat face turns across the fan (the swords of one clump overlap face to face).
                let normal = normalize3(cross3(fanDir, yAxis)) * (k % 2 == 0 ? 1 : -1)
                let length = size * (0.72 + 0.26 * rng.unit()) * (1 - 0.18 * abs(spread))
                let width = length * (0.085 + 0.02 * rng.unit())
                let leaf = GardenPlantMesh.FleshyLeaf(
                    base: centre + fanDir * (spread * width * 0.9), yAxis: yAxis, normal: normal,
                    length: length, width: width, thickness: max(0.003, width * 0.07), crown: 0.35,
                    tipThickness: 0.45, channel: 0.06, arch: 0.06 + 0.10 * rng.unit())
                let tone = FoliagePaint.tone(youth: 0, palette: palette, rng: &rng)
                let phase = rng.unit() * 2 * Double.pi
                let painted = GardenPlantMesh.fleshyLeafPainted(leaf, silhouette: snakePlantSwordRows,
                                                                tier: .hero, subdivisions: 1) { v, face in
                    FoliagePaint.snakePlantBand(palette, tone, v: v.v, u: v.u, phase: phase,
                                                margin: false)
                }
                out.leaves.append(painted)
            }
        }
        return out
    }

    // MARK: - Fern

    static func fernFoliage(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix) -> Foliage {
        let palette = PlantStyle.fern.foliagePalette
        let size = params.plantSize
        let count = leafCount(22, params.foliageDensity, minimum: 6)
        var out = Foliage()
        let start = rng.unit() * 2 * Double.pi
        for i in 0 ..< count {
            let t = count > 1 ? Double(i) / Double(count - 1) : 1   // 0 outer, oldest → 1 centre, newest
            let azimuth = start + Double(i) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.3
            let crown = params.potRadius * 0.16 * rng.unit().squareRoot()
            let base = Vec3(cos(azimuth) * crown, soilY + 0.005, sin(azimuth) * crown)
            let length = size * (1.15 - 0.40 * t) * (0.85 + rng.unit() * 0.30)
            let tilt = 1.25 - 0.95 * t + (rng.unit() - 0.5) * 0.2   // outer fronds arch out, inner stand
            let youth = smoothstep(0.72, 1.0, t)
            let tone = FoliagePaint.tone(youth: youth, palette: palette, rng: &rng)
            emitFernFrond(into: &out.leaves, base: base, azimuth: azimuth, tilt: tilt, length: length,
                          droop: 0.30 + 0.30 * (1 - t) + rng.unit() * 0.1, palette: palette, tone: tone,
                          rng: &rng)
        }
        return out
    }

    /// One Boston-fern frond: an arching rachis carrying opposite pairs of small keeled pinnae, the
    /// longest a third of the way up, tapering to nothing at the tip — the lanceolate plume. A pinna
    /// is eight triangles (a folded diamond, both faces): at the size a pinna is seen at, a rounded
    /// hero outline would be forty times the triangles for nothing the eye can read.
    static func emitFernFrond(into p: inout PaintedMesh, base: Vec3, azimuth: Double, tilt: Double,
                              length: Double, droop: Double, palette: FoliagePalette, tone: LeafTone,
                              rng: inout SplitMix) {
        let outDir = Vec3(cos(azimuth), 0, sin(azimuth))
        let up = Vec3(0, 1, 0)
        let reach = length * sin(tilt) * 0.95
        let rise = length * cos(tilt) + length * 0.25
        func rachis(_ v: Double) -> Vec3 {
            base + outDir * (reach * v) + up * (rise * sin(v * 1.4) * (1 - 0.15 * v) - droop * length * v * v)
        }
        let path = (0 ... 7).map { rachis(Double($0) / 7) }
        let stalk = FoliagePaint.mix3(palette.stalk, FoliagePaint.base(palette, tone, face: .upper), 0.35)
        p.append(paintedStalk(path, rBase: max(0.0018, length * 0.006), rTip: 0.0008, sides: 4, color: stalk))

        let pairs = 16
        var pinnae = PaintedMesh()
        for k in 0 ..< pairs {
            let v = 0.10 + 0.88 * Double(k) / Double(pairs - 1)
            let anchor = rachis(v)
            let tangent = normalize3(rachis(min(1, v + 0.01)) - rachis(max(0, v - 0.01)))
            var across = cross3(tangent, up)
            if len3(across) < 1e-6 { across = Vec3(-outDir.z, 0, outDir.x) }
            across = normalize3(across)
            let plane = normalize3(cross3(across, tangent))           // the frond's upper face
            // Lanceolate outline: longest at v ≈ 0.3, gone at the tip.
            let prof = pow(sin(Double.pi * min(1, v * 1.05)), 0.7) * (1 - 0.35 * v)
            let lp = length * 0.095 * prof * (0.9 + rng.unit() * 0.2)
            guard lp > length * 0.004 else { continue }
            let w = lp * 0.30
            for side in [1.0, -1.0] {
                // Pinnae angle forward, toward the frond tip, and droop a little at their ends.
                let dir = normalize3(across * (side * 0.94) + tangent * 0.34 - plane * 0.12)
                let wAx = normalize3(cross3(plane, dir)) * side
                let b = anchor
                let tip = b + dir * lp
                let mid = b + dir * (lp * 0.42) - plane * (w * 0.08)
                let l = b + dir * (lp * 0.40) + wAx * w + plane * (w * 0.25)
                let r = b + dir * (lp * 0.40) - wAx * w + plane * (w * 0.25)
                let n = plane
                let cTop = FoliagePaint.base(palette, tone, face: .upper)
                let cTip = FoliagePaint.mix3(cTop, palette.young * tone.value, 0.35)
                let cLow = FoliagePaint.base(palette, tone, face: .lower)
                for face in [LeafFace.upper, .lower] {
                    let o = face == .upper ? n : Vec3(-n.x, -n.y, -n.z)
                    let c0 = face == .upper ? cTop : cLow
                    let c1 = face == .upper ? cTip : cLow * 1.1
                    pinnae.addTriangle(b, l, mid, colors: c0, c0, c0, outward: o)
                    pinnae.addTriangle(b, mid, r, colors: c0, c0, c0, outward: o)
                    pinnae.addTriangle(mid, l, tip, colors: c0, c0, c1, outward: o)
                    pinnae.addTriangle(mid, tip, r, colors: c0, c1, c0, outward: o)
                }
            }
        }
        p.append(pinnae.smoothed(creaseDegrees: 50))
    }

    // MARK: - Succulent (echeveria rosette)

    static func succulentFoliage(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix) -> Foliage {
        let palette = PlantStyle.succulent.foliagePalette
        let up = Vec3(0, 1, 0)
        let R = params.plantSize * 0.42
        let n = leafCount(Double(GardenPlantMesh.echeveriaLeafCount) * 0.8, params.foliageDensity, minimum: 12)
        let phase = rng.unit() * 2 * Double.pi
        let centre = Vec3(0, soilY, 0)
        var out = Foliage()
        for i in 0 ..< n {
            let t = Double(i) / Double(max(1, n - 1))          // 0 outermost → 1 the heart
            let radial = GardenPlantMesh.rosetteRadial(axis: up, index: i, phase: phase)
            let pitch = 0.12 + 1.15 * pow(t, 1.6)
            let frame = GardenPlantMesh.rosetteLeafFrame(axis: up, radial: radial, pitch: pitch)
            let length = R * (1.02 - 0.62 * t) * (0.94 + rng.unit() * 0.12)
            let width = length * (0.52 + 0.08 * t)
            let leaf = GardenPlantMesh.FleshyLeaf(
                base: centre + radial * (R * 0.06 * (1 - t)) + up * (R * (0.05 + 0.22 * t)),
                yAxis: frame.yAxis, normal: frame.normal, length: length, width: width,
                thickness: width * 0.24, crown: 0.5, tipThickness: 0.55,
                channel: 0.03 + 0.10 * t, arch: -(0.12 + 0.30 * t))
            let tone = FoliagePaint.tone(youth: smoothstep(0.55, 1.0, t), palette: palette, rng: &rng)
            // The blush: outer, sun-struck leaves blush further down their length.
            let blushFrom = 0.62 + 0.25 * t
            let tier: LeafSilhouette.Tier = t > 0.55 ? .mid : .hero
            let painted = GardenPlantMesh.fleshyLeafPainted(leaf, silhouette: .echeveriaLeaf, tier: tier,
                                                            subdivisions: t > 0.55 ? 1 : 2) { v, face in
                let body = FoliagePaint.base(palette, tone, face: face == .lower ? .lower : .upper)
                let blush = smoothstep(blushFrom, 1.0, v.v) * (face == .lower ? 0.6 : 0.9)
                return FoliagePaint.clampColor(FoliagePaint.mix3(body, palette.rib, blush))
            }
            out.leaves.append(painted)
        }
        return out
    }
}
