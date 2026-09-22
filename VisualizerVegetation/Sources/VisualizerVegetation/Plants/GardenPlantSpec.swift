import simd
import VisualizerMaterials

/// The parameter spine for one in-ground garden plant — a shrub or flowering clump planted
/// straight into the soil (NOT in a pot, unlike `PlacedPottedPlant`). Reuses the potted plant's
/// generative foliage vocabulary (`PlantStyle` + the Forest-quality `PlantLeafCard` blades) but
/// grows the stem + leaves directly from the ground: no vessel, no soil disc.
///
/// **Single source of truth:** every dimension of the plant derives from these numbers — the
/// foliage from `style`/`size`/`foliageDensity`/`seed`. There is never a second width/height field.
public struct GardenPlantSpec: Equatable, Hashable, Sendable, Codable {
    /// The generative foliage / bouquet style (shared with the potted plant — a shrub reads as a
    /// dense fern/succulent rosette, a flower bed as the `.flowers` bloom clump).
    public var style: PlantStyle
    /// Overall plant height — the foliage rising from the ground (metres). ~0.15…1.5 m. The
    /// default (0.8 m) is a knee-high shrub/clump: a 0.5 m default read as a speck at yard
    /// distance (DH-0593), so a fresh drop is now a plant you can actually see across the lot.
    public var size: Double
    /// Leaf-density multiplier (0.5 sparse → 2.0 lush; 1.0 default). Scales the per-style leaf count.
    public var foliageDensity: Float
    /// Deterministic variation seed — two plants of the same style but different seeds get distinct
    /// (but style-consistent) foliage. Never an unseeded `.random`.
    public var seed: UInt64
    /// A real botanical species (DH-0648). When set it WINS: the plant is built by that species'
    /// own growth-habit builder and wears its colours, and `style` is not read. `nil` = a generic
    /// `style` plant, which is every plant saved before species existed.
    public var species: GardenPlantSpecies?

    public init(style: PlantStyle = .fern, size: Double = 0.8,
                foliageDensity: Float = 1.0, seed: UInt64 = 1,
                species: GardenPlantSpecies? = nil) {
        self.style = style
        self.size = min(species?.maxSize ?? 1.5, max(0.15, size))
        self.foliageDensity = min(2.0, max(0.5, foliageDensity))
        self.seed = seed
        self.species = species
    }

    /// A species plant at that species' natural height.
    public init(species: GardenPlantSpecies, foliageDensity: Float = 1.0, seed: UInt64 = 1) {
        self.init(size: species.defaultSize, foliageDensity: foliageDensity, seed: seed, species: species)
    }

    /// What this plant is, as the one value the inspector picks over.
    public var kind: GardenPlantKind {
        get { species.map { .species($0) } ?? .style(style) }
        set {
            switch newValue {
            case .style(let s):   style = s; species = nil
            case .species(let s): species = s
            }
        }
    }

    // MARK: derived dimensions (metres) — the SINGLE-SOURCE cascade

    /// Nominal base radius the foliage fans around — a small "planting spot" the leaf petioles
    /// emerge from. Scales gently with size so a big shrub spreads a little wider at the base.
    public var baseRadius: Double { min(0.18, max(0.06, size * 0.14)) }

    /// Plan half-width of the footprint / pick box — the foliage spreads roughly this far. A
    /// species spreads by its own habit (a poppy mat is far wider than it is tall).
    public var planHalfWidth: Double { max(0.2, size * (species?.spreadPerHeight ?? 0.5)) }

    // MARK: render colours — species first, else the style's

    public var leafColor: Vec3 { species?.leafColor ?? style.leafColor }
    public var leafRoughness: Float { species?.leafRoughness ?? style.leafRoughness }
    /// The opaque stem part's albedo. A generic style keeps the neutral dark green-brown stalk.
    public var stemColor: Vec3 { species?.stemColor ?? Vec3(0.20, 0.24, 0.10) }

    /// The synthesized `PottedPlantParams` the shared `PottedPlantMesh` foliage builders read.
    /// Pot dims are placeholders — the garden-plant mesh never emits the vessel/soil; only the
    /// foliage/stem/bloom builders are used, and they read `plantSize`/`plantStyle`/`foliageDensity`/
    /// `seed`/`potRadius` (petiole reach). Single source: this is the ONE bridge to the potted spine.
    public var plantParams: PottedPlantShape {
        PottedPlantShape(potRadius: baseRadius, plantSize: size,
                         plantStyle: style, foliageDensity: foliageDensity, seed: seed)
    }

    // MARK: tolerant Codable
    private enum CodingKeys: String, CodingKey { case style, size, foliageDensity, seed, species }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(style: try c.decodeIfPresent(PlantStyle.self, forKey: .style) ?? .fern,
                  size: try c.decodeIfPresent(Double.self, forKey: .size) ?? 0.8,
                  foliageDensity: try c.decodeIfPresent(Float.self, forKey: .foliageDensity) ?? 1.0,
                  seed: try c.decodeIfPresent(UInt64.self, forKey: .seed) ?? 1,
                  species: try c.decodeIfPresent(GardenPlantSpecies.self, forKey: .species))
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(style, forKey: .style)
        try c.encode(size, forKey: .size)
        try c.encode(foliageDensity, forKey: .foliageDensity)
        try c.encode(seed, forKey: .seed)
        try c.encodeIfPresent(species, forKey: .species)
    }
}
