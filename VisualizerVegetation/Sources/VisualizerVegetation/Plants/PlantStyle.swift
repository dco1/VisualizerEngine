import simd
import VisualizerMaterials

/// The generative style of a potted plant's foliage. Drives leaf SHAPE, COUNT, and
/// ARRANGEMENT in `PottedPlantMesh` — one enum, five recognizable house-plants. Taste:
/// the specific silhouettes/palettes are neutral defaults, flagged for Danny.
public enum PlantStyle: String, Sendable, Equatable, Hashable, Codable, CaseIterable {
    case fiddleLeafFig   // tall single stem, a few big broad glossy leaves
    case monstera        // broad split-leaf fronds on arching petioles, mid-height
    case snakePlant      // stiff, thin, tall upright blades in a clustered rosette
    case fern            // many feathered, arching fronds from a low crown
    case succulent       // small, low, tightly-clustered fat rosette leaves
    case flowers         // a mixed cut-flower bouquet: upright stems, each topped with a bloom
    case driedSpray      // a dried arrangement: many wiry brown stems, sprays of tiny mauve blooms
    case christmasTree   // a potted fir: stacked drooping bough tiers, baubles, a spiral of string lights

    public var label: String {
        switch self {
        case .fiddleLeafFig: return "Fiddle-Leaf Fig"
        case .monstera:      return "Monstera"
        case .snakePlant:    return "Snake Plant"
        case .fern:          return "Fern"
        case .succulent:     return "Succulent"
        case .flowers:       return "Flowers"
        case .driedSpray:    return "Dried Flowers"
        case .christmasTree: return "Christmas Tree"
        }
    }

    /// True for a *cut-flower bouquet* style (blooms on stems) rather than a foliage plant. The
    /// natural vessel for a bouquet is a `.vase`, so choosing `.flowers` defaults the vessel to
    /// `.vase` (still independently overridable — a pot can hold flowers, a vase can hold foliage).
    public var isBouquet: Bool { self == .flowers || self == .driedSpray }

    /// True for the one style that carries a string of lights — the only style for which
    /// `PottedPlantParams.stringLightsOn` means anything, and the only plant a ⌘-click switches
    /// rather than jostles (the lamp / candle "primary state" convention).
    public var hasStringLights: Bool { self == .christmasTree }
}

/// The kind of *vessel* a plant sits in — orthogonal to the plant content. A `.pot` is the
/// classic tapered planter (foliage's natural home); a `.vase` is a taller, narrower flower vase
/// with an elegant bellied-then-necked silhouette (a bouquet's natural home). The two are
/// independently settable — the natural default when `.flowers` is chosen is `.vase`, but a pot
/// can hold flowers and a vase can hold foliage.
public enum VesselKind: String, Sendable, Equatable, Hashable, Codable, CaseIterable {
    case pot     // classic tapered planter (base narrower than the rim)
    case vase    // taller, narrower flower vase: bulging belly → narrow neck → flared lip

    public var label: String {
        switch self {
        case .pot:  return "Pot"
        case .vase: return "Vase"
        }
    }
}

public extension PlantStyle {
    /// The natural size of this plant, in metres — the ONE table of per-style dimensions, read
    /// by `PottedPlantParams.init` as its defaults. A succulent is a fistful of leaves in a
    /// 9 cm pot and a fiddle-leaf fig is chest-high; giving all six the same 0.7 m foliage made
    /// every drop the wrong object, and made the tabletop plants (which `stackingRole` lifts
    /// onto furniture) unreachable without hand-shrinking the sliders first.
    ///
    /// The three big foliage plants land above `PottedPlantParams.tabletopHeightLimit` and are
    /// floor obstacles; the fern, the succulent, and a vase of cut flowers land below it and
    /// ride a table — which is the classification a person would make looking at them.
    /// (Taste: real-world-plausible defaults, flagged for Danny — every one is still a slider.)
    public var defaultDimensions: (potRadius: Double, potHeight: Double, plantSize: Double) {
        switch self {
        case .fiddleLeafFig: return (0.16, 0.28, 0.95)   // ≈1.23 m — a floor plant
        case .monstera:      return (0.16, 0.26, 0.78)   // ≈1.04 m — a floor plant
        case .snakePlant:    return (0.13, 0.24, 0.72)   // ≈0.96 m — a floor plant
        case .fern:          return (0.13, 0.19, 0.44)   // ≈0.63 m — rides a table/stand
        case .succulent:     return (0.07, 0.09, 0.20)   // ≈0.29 m — a desk succulent
        // A bouquet's vessel is a vase, whose height is 1.8× `potHeight` (`vesselHeight`), so
        // 0.16 here is a ~29 cm vase: ≈0.59 m of flowers total — a centrepiece, not a floor vase.
        case .flowers:       return (0.075, 0.16, 0.30)
        case .driedSpray:    return (0.055, 0.15, 0.34)   // ≈0.49 m — a counter/credenza vase
        // A living-room tree in a big planter: ≈1.70 m, a floor piece. `plantSize` tops out at the
        // shared 1.5 m clamp, so the tallest tree this style makes is ≈1.8 m with a tall pot.
        case .christmasTree: return (0.20, 0.30, 1.40)
        }
    }

    /// The base leaf albedo for this style (linear-ish RGB). Distinct greens per style give the
    /// five plants a readable identity. Seeded per-leaf jitter is applied in the mesh. Taste
    /// (neutral defaults — flagged for Danny).
    public var leafColor: Vec3 {
        // Brightened ~35% from the first pass — the realism audit read the leaves as
        // "dark construction-paper cutouts"; front-lit houseplant foliage is far
        // livelier than forest-canopy green (the SSS backlit glow rides on top).
        //
        // WARMED off cyan (2026-07-11): the earlier pass had blue ≥ red in 5 of 6 styles,
        // which reads as an unnatural blue/cyan-green cardboard cutout (a real-Metal capture
        // measured monstera leaves at R47.7 < B49.8). Natural front-lit foliage is a YELLOW-
        // green — red sits ABOVE blue. Every style now keeps green clearly dominant but with
        // R > B by a healthy margin (`PlantStyle.leafColorIsWarmGreen` gate). Same family of
        // fix as the grass-blade / lawn de-saturation.
        switch self {
        case .fiddleLeafFig: return Vec3(0.18, 0.44, 0.10)   // deep glossy green
        case .monstera:      return Vec3(0.22, 0.50, 0.13)   // rich mid green
        case .snakePlant:    return Vec3(0.26, 0.47, 0.16)   // grey-green blade
        case .fern:          return Vec3(0.28, 0.57, 0.15)   // bright fresh green
        case .succulent:     return Vec3(0.41, 0.56, 0.28)   // pale sage green
        case .flowers:       return Vec3(0.24, 0.50, 0.13)   // bouquet stem/leaf green
        case .driedSpray:    return Vec3(0.40, 0.46, 0.20)   // the few dried olive leaves
        case .christmasTree: return Vec3(0.12, 0.27, 0.09)   // deep fir-needle green, still warm (R > B)
        }
    }

    /// A leaf albedo reads as a natural warm (yellow-)green when green is clearly the dominant
    /// channel AND red sits above blue — the opposite of the cyan/blue-green tell this fix
    /// removed. Deterministic hue gate (`PottedPlantMeshTests` fails if any style drifts cool).
    public var leafColorIsWarmGreen: Bool {
        let c = leafColor
        return c.y > c.x && c.y > c.z && c.x > c.z
    }

    /// Leaf-surface roughness for the render instance — waxy broadleaf houseplants
    /// (fig/monstera/succulent) are visibly GLOSSY (the audit's matte cards read as
    /// paper); ferns stay matte, blades in between. Single source with `leafColor`.
    public var leafRoughness: Float {
        switch self {
        case .fiddleLeafFig, .monstera, .succulent: return 0.30
        case .snakePlant:                           return 0.42
        case .fern, .flowers:                       return 0.55
        case .driedSpray:                           return 0.70   // papery, dead-matte
        case .christmasTree:                        return 0.78   // a mass of needles scatters — no sheen
        }
    }
}

/// **What a potted plant's GEOMETRY reads** — the pot and plant dimensions, style and seed — so the
/// generators in `PottedPlantMesh` take any value that can answer them. A host's own stored record
/// (with its materials, its persistence, its UI state) conforms; the generators never see the rest.
public protocol PottedPlantGeometry {
    var potRadius: Double { get }
    var potHeight: Double { get }
    var plantSize: Double { get }
    var plantStyle: PlantStyle { get }
    var foliageDensity: Float { get }
    var seed: UInt64 { get }
    /// The vessel the mesh is built with.
    var resolvedVessel: VesselKind { get }
}

/// A plain value of `PottedPlantGeometry` — what a garden plant synthesises to borrow the potted
/// plant's foliage, stem and bouquet builders, and what any host without its own record passes.
public struct PottedPlantShape: PottedPlantGeometry, Sendable, Equatable, Hashable {
    public var potRadius: Double
    public var potHeight: Double
    public var plantSize: Double
    public var plantStyle: PlantStyle
    public var foliageDensity: Float
    public var seed: UInt64
    public var resolvedVessel: VesselKind
    /// Dimensions default to the style's natural size, clamped to the same ranges a stored potted
    /// plant uses (radius 0.05–0.25 m, height 0.08–0.50 m, plant 0.05–1.5 m, density 0.5–2).
    public init(potRadius: Double? = nil, potHeight: Double? = nil, plantSize: Double? = nil,
                plantStyle: PlantStyle = .fiddleLeafFig, foliageDensity: Float = 1.0, seed: UInt64 = 1,
                vessel: VesselKind = .pot) {
        let d = plantStyle.defaultDimensions
        self.potRadius = min(0.25, max(0.05, potRadius ?? d.potRadius))
        self.potHeight = min(0.50, max(0.08, potHeight ?? d.potHeight))
        self.plantSize = min(1.5, max(0.05, plantSize ?? d.plantSize))
        self.plantStyle = plantStyle
        self.foliageDensity = min(2.0, max(0.5, foliageDensity))
        self.seed = seed
        self.resolvedVessel = vessel
    }
}
