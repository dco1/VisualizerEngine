import simd
import VisualizerMaterials

/// A REAL botanical species an in-ground garden plant can be (DH-0648) — as opposed to a generic
/// `PlantStyle`, which is a houseplant silhouette shared with the potted plant.
///
/// **Why a separate axis and not more `PlantStyle` cases.** A `PlantStyle` is a leaf silhouette on
/// the potted plant's arranger (a rosette, a stem spiral, a bouquet). A species is a whole GROWTH
/// HABIT — the California poppy's sprawling mat of thread-leaves carrying flowers at four stages at
/// once, the Matilija poppy's tall multi-stemmed clump holding a few huge flowers on bare stems
/// above its foliage — and none of that fits the potted arranger's knobs. Each species here owns its
/// own builder (`GardenPlantMesh+CaliforniaPoppy`, `+MatilijaPoppy`) authored on the shared blade
/// constructor (`VisualizerVegetation.LeafConstructor`), and its own leaf/stem/petal colours.
///
/// The cacti (DH-0649) are the case for the axis at its strongest: the golden barrel is a ribbed
/// body with no leaves at all, bunny ears a branching chain of solid pads (`PadConstructor`). For
/// both, the opaque `stem` part is the fleshy body and the `foliage` part is what stands in for
/// leaves — spines, glochid tufts — so `leafColor` is theirs.
///
/// The rosette succulents (DH-0797) — aloe, echeveria, century plant — are thick fleshy leaves, each a
/// SOLID shaped from a `PadConstructor` pad (`GardenPlantMesh+Rosette`). The leaves are the opaque
/// `stem` part; the `foliage` part is the aloe's and agave's teeth and spines, and the echeveria's
/// blushed leaf tips, so `leafColor` is theirs.
///
/// A species is optional on `GardenPlantSpec`: `nil` keeps the plant a generic `style`.
public enum GardenPlantSpecies: String, Sendable, Equatable, Hashable, Codable, CaseIterable {
    /// *Eschscholzia californica* — the state flower. Bedding scale: a low sprawling mat.
    case californiaPoppy
    /// *Romneya coulteri* — "the fried egg flower". Shrub scale: a tall multi-stemmed clump.
    case matilijaPoppy
    /// *Echinocactus grusonii* — a ribbed, golden-spined sphere that becomes a barrel with age.
    case goldenBarrel
    /// *Opuntia microdasys* 'Monstrosa' — warped branching pads dotted with glochid tufts.
    case bunnyEars
    /// *Bougainvillea glabra* — a thorny arching vine whose colour is papery bracts, not petals.
    case bougainvillea
    /// *Strelitzia reginae* — paddle leaves in fans, and the crane's-head flower on a bare stalk.
    case birdOfParadise
    /// *Ceanothus* 'Ray Hartman' — a woody shrub under dense puffballs of true-blue florets.
    case californiaLilac
    /// *Aloe arborescens* — recurved toothed rosettes on woody stems, under red-orange racemes.
    case aloe
    /// *Echeveria elegans* — a tight pale spoon-leaf rosette on the soil, ringed by its chicks.
    case echeveria
    /// *Agave americana* — a huge rigid blue-grey rosette armed with hooked teeth and spines.
    case centuryPlant
    /// *Yucca gigantea* — the giant yucca: bare forking trunks off a swollen foot, each ending in a
    /// head of sword leaves, some carrying a tall panicle of edible white bells. A TREE, not a shrub.
    case giantYucca

    public var label: String {
        switch self {
        case .californiaPoppy: return "California Poppy"
        case .matilijaPoppy:   return "Matilija Poppy"
        case .goldenBarrel:    return "Golden Barrel Cactus"
        case .bunnyEars:       return "Bunny Ears Cactus"
        case .bougainvillea:   return "Bougainvillea"
        case .birdOfParadise:  return "Bird of Paradise"
        case .californiaLilac: return "California Lilac"
        case .aloe:            return "Aloe"
        case .echeveria:       return "Echeveria"
        case .centuryPlant:    return "Century Plant"
        case .giantYucca:      return "Giant Yucca"
        }
    }

    public var scientificName: String {
        switch self {
        case .californiaPoppy: return "Eschscholzia californica"
        case .matilijaPoppy:   return "Romneya coulteri"
        case .goldenBarrel:    return "Echinocactus grusonii"
        case .bunnyEars:       return "Opuntia microdasys 'Monstrosa'"
        case .bougainvillea:   return "Bougainvillea glabra"
        case .birdOfParadise:  return "Strelitzia reginae"
        case .californiaLilac: return "Ceanothus 'Ray Hartman'"
        case .aloe:            return "Aloe arborescens"
        case .echeveria:       return "Echeveria elegans"
        case .centuryPlant:    return "Agave americana"
        case .giantYucca:      return "Yucca gigantea"
        }
    }

    /// The height a fresh plant of this species is given (metres). A poppy mat is knee-low; a
    /// Matilija clump is as tall as the spec allows (a real one reaches 2.4 m — the shared 1.5 m
    /// clamp is the ceiling here). A garden-centre golden barrel is a ball about knee height; the
    /// size is also its age (a tall one is an old barrel). A free-standing bougainvillea fountain of
    /// canes is about head height, a bird of paradise clump waist-to-chest high, and a Ceanothus is
    /// a big shrub capped by the same 1.5 m clamp as the Matilija. A candelabra aloe that has
    /// branched is about a metre, raceme tips included; an echeveria is a hand-sized rosette at the
    /// spec's 15 cm floor; a century plant is about 1.2 m of leaf.
    /// The tallest this species may be drawn (metres). Every bedding plant and shrub shares the
    /// 1.5 m ceiling; the giant yucca is a tree and gets a tree's.
    public var maxSize: Double { self == .giantYucca ? 7.0 : 1.5 }

    public var defaultSize: Double {
        switch self {
        case .californiaPoppy: return 0.35
        case .matilijaPoppy:   return 1.5
        case .goldenBarrel:    return 0.45
        case .bunnyEars:       return 0.5
        case .bougainvillea:   return 1.4
        case .birdOfParadise:  return 1.2
        case .californiaLilac: return 1.5
        case .aloe:            return 1.0
        case .echeveria:       return 0.15
        case .centuryPlant:    return 1.2
        case .giantYucca:      return 4.5
        }
    }

    /// Plan half-spread as a fraction of height. A poppy SPRAWLS — a mat far wider than it is tall;
    /// the Matilija clump is roughly as wide as half its height each side. A barrel's spread is its
    /// body plus the spine halo; bunny ears branch out about as far. Bougainvillea canes arch out and
    /// hang well past their crown; the strelitzia fans and the Ceanothus dome sit near half. The
    /// century plant is where spread has to carry the scale the 1.5 m `size` clamp cannot: its
    /// rosette is over twice as wide as it is tall. An echeveria's chicks ring out past its rosette.
    public var spreadPerHeight: Double {
        switch self {
        case .californiaPoppy: return 0.85
        case .matilijaPoppy:   return 0.45
        case .goldenBarrel:    return 0.62
        case .bunnyEars:       return 0.62
        case .bougainvillea:   return 0.80
        case .birdOfParadise:  return 0.52
        case .californiaLilac: return 0.58
        case .aloe:            return 0.62
        case .echeveria:       return 0.95
        case .centuryPlant:    return 1.15
        case .giantYucca:      return 0.36
        }
    }

    /// Foliage albedo. Both poppies are GLAUCOUS — a waxy grey/blue-green bloom on the leaf — but it
    /// stays a warm-leaning grey-green (red above blue, the `leafColorIsWarmGreen` rule): a truly
    /// cyan leaf measured as a construction-paper cutout under this renderer. A cactus's foliage part
    /// is its spines (gold) or glochid tufts (tan). Bougainvillea is a plain mid-green; the
    /// strelitzia a deep waxy blue-green (still red above blue); Ceanothus a dark glossy evergreen.
    /// A rosette succulent's foliage part is not its leaves: the aloe's soft pale teeth, the
    /// echeveria's pink-blushed tips, the century plant's dark brown teeth and spines.
    public var leafColor: Vec3 {
        switch self {
        case .californiaPoppy: return Vec3(0.33, 0.46, 0.30)
        case .matilijaPoppy:   return Vec3(0.43, 0.50, 0.41)
        case .goldenBarrel:    return Vec3(0.80, 0.62, 0.20)
        case .bunnyEars:       return Vec3(0.80, 0.64, 0.34)
        case .bougainvillea:   return Vec3(0.30, 0.44, 0.19)
        case .birdOfParadise:  return Vec3(0.25, 0.37, 0.24)
        case .californiaLilac: return Vec3(0.16, 0.26, 0.11)
        case .aloe:            return Vec3(0.74, 0.72, 0.52)
        case .echeveria:       return Vec3(0.76, 0.50, 0.52)
        case .centuryPlant:    return Vec3(0.28, 0.19, 0.12)
        case .giantYucca:      return Vec3(0.20, 0.30, 0.13)     // the sword leaves themselves
        }
    }

    /// Leaf roughness. The glaucous wax scatters — neither poppy leaf is glossy. Barrel spines are
    /// hard and catch a glint; glochid tufts are fuzz. Strelitzia and Ceanothus leaves are glossy.
    /// Aloe teeth are soft tissue; the echeveria's blush is under the same farina as the rest of the
    /// leaf; agave spines are hard.
    public var leafRoughness: Float {
        switch self {
        case .californiaPoppy: return 0.55
        case .matilijaPoppy:   return 0.62
        case .goldenBarrel:    return 0.42
        case .bunnyEars:       return 0.85
        case .bougainvillea:   return 0.55
        case .birdOfParadise:  return 0.36
        case .californiaLilac: return 0.34
        case .aloe:            return 0.60
        case .echeveria:       return 0.82
        case .centuryPlant:    return 0.45
        case .giantYucca:      return 0.5
        }
    }

    /// The opaque stem albedo. The poppy's stem is smooth waxy blue-green; the Matilija's `stem` part
    /// is its woody grey base (its green upper stems ride a colour group). A cactus's stem part is
    /// its whole fleshy body: the barrel a deep green, the bunny-ears pads a pale blue-green. The
    /// bougainvillea and Ceanothus stem parts are their grey-brown old wood (young growth rides a
    /// colour group); the strelitzia's is its green petioles and bare flower stalks. A rosette
    /// succulent's stem part is its fleshy leaves: the aloe a solid grey-green, the echeveria a pale
    /// powdery blue-grey, the century plant a chalky blue-grey-green.
    public var stemColor: Vec3 {
        switch self {
        case .californiaPoppy: return Vec3(0.34, 0.46, 0.31)
        case .matilijaPoppy:   return Vec3(0.40, 0.38, 0.33)
        case .goldenBarrel:    return Vec3(0.17, 0.29, 0.12)
        case .bunnyEars:       return Vec3(0.38, 0.47, 0.36)
        case .bougainvillea:   return Vec3(0.26, 0.22, 0.17)
        case .birdOfParadise:  return Vec3(0.26, 0.34, 0.22)
        case .californiaLilac: return Vec3(0.24, 0.21, 0.18)
        case .aloe:            return Vec3(0.35, 0.43, 0.33)
        case .echeveria:       return Vec3(0.52, 0.60, 0.57)
        case .centuryPlant:    return Vec3(0.42, 0.50, 0.48)
        case .giantYucca:      return Vec3(0.17, 0.14, 0.11)     // bare grey-brown trunks
        }
    }
}

/// What a garden plant IS, as one value — a generic style or a real species. The inspector's one
/// "Plant" picker selects over this, so the two axes can never be set to disagree from the UI.
public enum GardenPlantKind: Sendable, Hashable {
    case style(PlantStyle)
    case species(GardenPlantSpecies)

    public var label: String {
        switch self {
        case .style(let s):   return s.label
        case .species(let s): return s.label
        }
    }

    /// Every choice, generic styles first.
    public static var allCases: [GardenPlantKind] {
        PlantStyle.allCases.map { .style($0) } + GardenPlantSpecies.allCases.map { .species($0) }
    }
}
