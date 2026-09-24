import Foundation
import simd
import VisualizerMaterials

/// Mesh generator for an in-ground garden plant (`PlacedGardenPlant`). Pure geometry. A generic
/// STYLE reuses the potted plant's growth (`PottedPlantMesh.foliage` / `stemMesh` / `bouquetMesh`,
/// all keyed by a synthesized `PottedPlantParams`) but grows it straight from the ground: there is
/// **no vessel and no soil disc** — that is the whole difference from a potted plant, and this file
/// never re-derives that geometry. A real SPECIES builds by its own habit (the `+Species` files).
///
/// **Local space:** X/Z plan (centred on the stem), Y up from the ground (y = 0 = ground). The
/// render bridge seats it at the plant's world spot. Deterministic in `(seed, spec)`.
public enum GardenPlantMesh {

    /// The plant's mesh parts, split so each renders with the right treatment: `stem` is the
    /// opaque woody stalk (vertex alpha 1); `foliage` is the GREEN leaf cards (tagged alpha-0 for
    /// thin-sheet SSS by the bridge — this is an OUTDOOR plant, so it keeps the grass/tree
    /// backlight, unlike an interior potted plant); `blooms` are the per-colour petal/center groups
    /// for the `.flowers` style (empty otherwise).
    public struct Parts: Sendable {
        public var stem: Mesh3
        public var foliage: Mesh3
        /// One albedo per `foliage` vertex when the generator painted the leaves (a generic style
        /// grows the potted plant's painted foliage); empty for a species, whose foliage part is its
        /// own single-colour detail (teeth, spines, blush) — the bridge then stamps `leafColor`.
        public var foliageColors: [Vec3]
        public var blooms: [PottedPlantMesh.ColoredGroup]
        public init(stem: Mesh3, foliage: Mesh3, foliageColors: [Vec3] = [],
                    blooms: [PottedPlantMesh.ColoredGroup] = []) {
            precondition(foliageColors.isEmpty || foliageColors.count == foliage.vertexCount,
                         "painted foliage needs one colour per vertex")
            self.stem = stem; self.foliage = foliage; self.foliageColors = foliageColors; self.blooms = blooms
        }
    }

    public static func parts(spec: GardenPlantSpec) -> Parts {
        // A real species builds by its own growth habit, not the potted plant's arranger (DH-0648).
        switch spec.species {
        case .californiaPoppy?: return californiaPoppyParts(spec: spec)
        case .matilijaPoppy?:   return matilijaPoppyParts(spec: spec)
        case .goldenBarrel?:    return goldenBarrelParts(spec: spec)
        case .bunnyEars?:       return bunnyEarsParts(spec: spec)
        case .bougainvillea?:   return bougainvilleaParts(spec: spec)
        case .birdOfParadise?:  return birdOfParadiseParts(spec: spec)
        case .californiaLilac?: return californiaLilacParts(spec: spec)
        case .aloe?:            return aloeParts(spec: spec)
        case .echeveria?:       return echeveriaParts(spec: spec)
        case .centuryPlant?:    return centuryPlantParts(spec: spec)
        case .giantYucca?:      return giantYuccaParts(spec: spec)
        case nil:               break
        }
        let params = spec.plantParams
        // Seed the RNG exactly like the potted plant so the SAME spec yields the SAME arrangement.
        var rng = PottedPlantMesh.SplitMix(spec.seed &* 0x2545F4914F6CDD1D &+ 0x9E3779B97F4A7C15)
        let soilY = 0.0   // rises straight from the ground — no vessel/soil disc

        if params.plantStyle.isBouquet {
            // A flowering clump: green stems + vivid blooms, rising from the soil.
            let bouquet = PottedPlantMesh.bouquetMesh(params: params, soilY: soilY, rng: &rng)
            return Parts(stem: Mesh3(), foliage: bouquet.greens, blooms: bouquet.blooms)
        }

        // Foliage plant: a woody stalk + the potted plant's own painted leaves, grown from the
        // ground instead of a pot — the same habit and the same paint either way.
        var stem = PottedPlantMesh.stemMesh(params: params, baseY: soilY)
        let grown = PottedPlantMesh.foliage(params: params, soilY: soilY, rng: &rng)
        stem.append(grown.woody)
        return Parts(stem: stem, foliage: grown.mesh, foliageColors: grown.colors)
    }

    /// The whole plant as ONE mesh (stem + leaves/blooms) — for the auditors, which check the
    /// merged geometry. (The bridge uses `parts` for the per-group materials + SSS flag.)
    public static func mesh(spec: GardenPlantSpec) -> Mesh3 {
        let p = parts(spec: spec)
        var m = p.stem
        m.append(p.foliage)
        for g in p.blooms { m.append(g.mesh) }
        return m
    }
}
