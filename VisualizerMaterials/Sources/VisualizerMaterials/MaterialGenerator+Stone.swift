import Foundation

/// The figure a natural-stone slab carries — the two ways a polished stone is patterned.
public enum StoneFigure: String, CaseIterable, Codable, Sendable, Hashable {
    /// Marble's domain-warped veins: a pale matrix crossed by trains of thin veins.
    case veined
    /// A BRECCIA (the "rosso" quartzites, Breccia Pernice, Rainforest): angular broken clasts of
    /// one or two colours cemented by a pale mineral matrix, with dark seams along some joints.
    case breccia

    public var displayName: String {
        switch self {
        case .veined:  return "Veined"
        case .breccia: return "Breccia"
        }
    }
}

/// A coloured natural stone: the figure plus the four mineral colours (linear albedo). Before this
/// existed the library stones came in ONE colourway each and the only lever was `MaterialGrade` —
/// which can lighten, warm or desaturate a white marble but can never make it a rust-and-pink
/// quartzite (Danny's warm-kitchen reference, 2026-10-02).
public struct StoneParams: Equatable, Hashable, Sendable, Codable {
    public var figure: StoneFigure
    /// The pale cement / matrix between clasts (breccia) or the ground of the slab (veined).
    public var matrix: Vec3
    /// The dominant clast colour (breccia) — unused by `.veined`.
    public var primary: Vec3
    /// The second clast colour, mixed per clast with `primary` (breccia) — unused by `.veined`.
    public var secondary: Vec3
    /// The dark seams / veins.
    public var vein: Vec3
    public var seed: UInt64

    public init(figure: StoneFigure = .breccia,
                matrix: Vec3 = StoneParams.rosso.matrix, primary: Vec3 = StoneParams.rosso.primary,
                secondary: Vec3 = StoneParams.rosso.secondary, vein: Vec3 = StoneParams.rosso.vein,
                seed: UInt64 = 17) {
        self.figure = figure; self.matrix = matrix; self.primary = primary
        self.secondary = secondary; self.vein = vein; self.seed = seed
    }

    private init(raw figure: StoneFigure, _ m: Vec3, _ p: Vec3, _ s: Vec3, _ v: Vec3) {
        self.figure = figure; matrix = m; primary = p; secondary = s; vein = v; seed = 17
    }

    /// Rosso quartzite: rust and dusty-pink clasts in a cream cement, umber seams.
    public static let rosso = StoneParams(raw: .breccia, Vec3(0.66, 0.44, 0.27), Vec3(0.33, 0.080, 0.022),
                                          Vec3(0.52, 0.17, 0.090), Vec3(0.075, 0.026, 0.010))
    /// Calacatta-like: warm white ground, grey-umber veins.
    public static let calacatta = StoneParams(raw: .veined, Vec3(0.84, 0.82, 0.78), Vec3(0.70, 0.66, 0.60),
                                              Vec3(0.62, 0.58, 0.52), Vec3(0.30, 0.27, 0.25))
    /// Verde: deep green breccia with pale veins.
    public static let verde = StoneParams(raw: .breccia, Vec3(0.55, 0.60, 0.52), Vec3(0.05, 0.12, 0.08),
                                          Vec3(0.10, 0.20, 0.14), Vec3(0.02, 0.04, 0.03))

    /// The named colourways a picker offers.
    public static let presets: [(name: String, params: StoneParams)] = [
        ("Rosso Quartzite", rosso), ("Calacatta", calacatta), ("Verde", verde),
    ]

    private enum CodingKeys: String, CodingKey { case figure, matrix, primary, secondary, vein, seed }

    /// Tolerant decode: any colour missing falls back to the rosso colourway, so a later field
    /// is not a migration.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = StoneParams.rosso
        figure = try c.decodeIfPresent(StoneFigure.self, forKey: .figure) ?? d.figure
        matrix = try c.decodeIfPresent(Vec3.self, forKey: .matrix) ?? d.matrix
        primary = try c.decodeIfPresent(Vec3.self, forKey: .primary) ?? d.primary
        secondary = try c.decodeIfPresent(Vec3.self, forKey: .secondary) ?? d.secondary
        vein = try c.decodeIfPresent(Vec3.self, forKey: .vein) ?? d.vein
        seed = try c.decodeIfPresent(UInt64.self, forKey: .seed) ?? d.seed
    }
}

extension MaterialGenerator {

    /// A coloured natural stone (`StoneParams`). `.veined` is the marble generator in the given
    /// colours; `.breccia` is `brecciaStone`.
    public static func naturalStone(size: Int = MaterialGenerator.bakeSize, params p: StoneParams = .rosso) -> MaterialChannels {
        switch p.figure {
        case .veined:  return marble(size: size, seed: p.seed, base: p.matrix, veinColor: p.vein)
        case .breccia: return brecciaStone(size: size, params: p)
        }
    }

    /// **Breccia** — rock that was shattered and re-cemented. Three things make it read as stone
    /// rather than as camouflage or terrazzo, and each is a term below:
    ///
    /// 1. **Clasts are ANGULAR and of very different sizes.** A Voronoi tessellation gives the
    ///    straight-edged broken fragments; a domain warp bends the straight joins just enough to
    ///    stop them reading as a mosaic, and a second, finer tessellation shatters some clasts into
    ///    crush zones of small chips (a breccia is fragments at every scale, not one cell size).
    /// 2. **Each clast is ONE colour, with its own internal mottle.** The colour is picked per
    ///    cell (between primary and secondary, sometimes bleached toward the matrix), and the
    ///    clast carries a crystalline fbm and a few hairline fractures inside it — a flat-filled
    ///    cell is the terrazzo tell.
    /// 3. **The cement is a band of varying width, and only SOME joints carry a dark seam.** A
    ///    uniform outline on every cell is a stained-glass leading; real seams come and go.
    ///
    /// Polished: low roughness, the clearcoat lobe marble takes, a faint relief where the softer
    /// cement polishes below the clasts.
    public static func brecciaStone(size: Int = MaterialGenerator.bakeSize, params p: StoneParams) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .stone)
        let seed = p.seed
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)

                // 1. Ragged, multi-scale tessellation. A two-frequency warp tears the straight
                //    Voronoi joins into broken edges; three lattices (big / medium / chips) are
                //    chosen region by region, so the slab has a few large clasts, fields of
                //    medium ones and shattered crush zones — fragments at every scale.
                let w1x = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: seed ^ 0x11) - 0.5
                let w1y = Noise.fbmTiled(u, v, baseCells: 3, octaves: 3, seed: seed ^ 0x23) - 0.5
                let w2x = Noise.fbmTiled(u, v, baseCells: 14, octaves: 3, seed: seed ^ 0x15) - 0.5
                let w2y = Noise.fbmTiled(u, v, baseCells: 14, octaves: 3, seed: seed ^ 0x27) - 0.5
                let wu = u + 0.10 * w1x + 0.025 * w2x, wv = v + 0.10 * w1y + 0.025 * w2y
                let scale = Noise.fbmTiled(u, v, baseCells: 2, octaves: 3, seed: seed ^ 0x47)
                let cell: Noise.Cell
                let edgeScale: Double
                if scale < 0.46 {
                    cell = Noise.voronoiTiled(wu, wv, cells: 5, jitter: 0.95, seed: seed ^ 0x35); edgeScale = 1.6
                } else if scale < 0.60 {
                    cell = Noise.voronoiTiled(wu, wv, cells: 11, jitter: 0.95, seed: seed ^ 0x59); edgeScale = 0.9
                } else {
                    cell = Noise.voronoiTiled(wu, wv, cells: 26, jitter: 0.95, seed: seed ^ 0x6A); edgeScale = 0.45
                }
                let h = cell.cellId

                // 2. The clast: a point on the stone's own palette ramp (umber → rust → pink →
                //    cream), clouded inside, with a few dark mineral inclusions.
                let pick = Noise.unit(h), pick2 = Noise.unit(Noise.mix(h ^ 0x9E37))
                var clast = mix(p.primary, p.secondary, smoothstep(0.25, 0.75, pick))
                if pick2 > 0.88 { clast = mix(clast, p.matrix, 0.55 + 0.3 * (pick2 - 0.88) / 0.12) }   // bleached clasts
                if pick2 < 0.22 { clast = mix(clast, p.vein, 0.40 + pick2) }                                     // umber clasts
                let cloud = Noise.fbmTiled(wu, wv, baseCells: 5, octaves: 4, seed: seed ^ (h & 0xFFFF))
                let grain = Noise.fbmTiled(u, v, baseCells: 64, octaves: 2, seed: seed ^ 0x6B)
                clast = mix(clast, mix(p.primary, p.secondary, 1 - smoothstep(0.25, 0.75, pick)), smoothstep(0.55, 0.85, cloud) * 0.6)
                clast = clast * (0.70 + 0.55 * cloud + 0.32 * (grain - 0.5))
                let incl = smoothstep(0.80, 0.92, Noise.fbmTiled(u, v, baseCells: 9, octaves: 4, seed: seed ^ 0xB3))
                clast = mix(clast, p.vein, incl * 0.55)
                // Crisp cracks across the clast — thin, straight-ish (few octaves), cream-filled
                // with a dark lip, the healed fractures of a shattered rock.
                let crack = 1 - abs(Noise.fbmTiled(wu, wv, baseCells: 4, octaves: 2, seed: seed ^ 0x7D) * 2 - 1)
                let crackFill = smoothstep(0.975, 0.995, crack)
                clast = mix(clast, p.matrix, crackFill * 0.85)
                // A fine dark veinlet network, the iron staining that runs through the clasts.
                let vl = 1 - abs(Noise.fbmTiled(wu, wv, baseCells: 10, octaves: 3, seed: seed ^ 0xD7) * 2 - 1)
                clast = mix(clast, p.vein, smoothstep(0.965, 0.995, vl) * 0.65)

                // 3. Joints: mostly a hairline, sometimes a pool of pale cement, and a dark seam
                //    along some of them.
                let edge = (cell.f2 - cell.f1) * edgeScale
                let pool = pow(Noise.fbmTiled(u, v, baseCells: 6, octaves: 3, seed: seed ^ 0x8F), 2.2)
                let cementW = 0.010 + 0.12 * pool
                let cement = 1 - smoothstep(cementW * 0.45, cementW, edge)
                let seamOn = smoothstep(0.42, 0.62, Noise.fbmTiled(u, v, baseCells: 5, octaves: 2, seed: seed ^ 0xA1))
                let seam = (1 - smoothstep(0.003, 0.012, edge)) * seamOn
                let cementColor = mix(p.matrix, p.secondary, 0.25 * Noise.fbmTiled(u, v, baseCells: 10, octaves: 3, seed: seed ^ 0xC5))
                    * (0.90 + 0.18 * grain)
                var c = mix(clast, mix(cementColor, clast, 0.25), cement)
                c = mix(c, p.vein, seam * 0.9)

                ch.albedo[ch.idx(x, y)] = clampBand(c)
                ch.roughness[ch.idx(x, y)] = clamp01(0.10 + cement * 0.06 + seam * 0.05
                    + (Noise.fbmTiled(u, v, baseCells: 28, octaves: 2, seed: seed ^ 0x4E) - 0.5) * 0.04)
                ch.height[ch.idx(x, y)] = clamp01(0.55 - cement * 0.10 - seam * 0.15 + (grain - 0.5) * 0.04)
            }
        }
        ch.clearcoat = 0.30                        // polished, the lobe marble takes
        ch.deriveNormals(strength: 2.0)
        addMicroDetail(&ch, seed: seed ^ 0xF4, baseCells: 72, strength: 0.45)
        return ch
    }
}
