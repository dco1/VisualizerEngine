import simd
import VisualizerMaterials

/// Everything about ONE species' blade, as a value.
///
/// This is the "species as data" unit. Before it, a species' blade was described in two unrelated
/// places in two unrelated shapes — `PottedPlantMesh.FoliagePlan` (15 fields, `switch` over
/// `PlantStyle`) and `ForestTreeGeometry.TreeLook` (24 fields, `switch` over `Species`) — each
/// carrying its own idea of blade size, its own hardcoded fold, and, in the tree's case, a THIRD
/// switch mapping its species onto a silhouette. Adding a plant meant editing a growing switch in
/// a shared file, and adding it to only one of the two systems was the default outcome.
///
/// A `BladeDescription` is just written down. The catalog below is `static let`s, so a new species
/// is a new constant — nothing that already compiles has to change to admit it.
///
/// **What this deliberately does NOT cover: arrangement.** Where a species puts its blades — a
/// basal rosette, a single spiralling stem, a compound frond, a recursive branch armature — is a
/// genuinely different topology per growth habit, not a field bag, and the two shipped arrangers
/// (`PottedPlantMesh.foliageMesh` and `ForestTreeGeometry.emitLeaves`) are real, tuned code with
/// gates on them. Folding those into one arranger is the next step, not this one; writing a third
/// arranger here to sit beside them would recreate the exact duplication this module exists to
/// remove. See `LeafArrangement` for the shape that work should take.
public struct BladeDescription: Sendable, Hashable {

    /// The outline.
    public var silhouette: LeafSilhouette

    /// Width as a fraction of length. A blade is authored at unit length and this sets how broad
    /// it is; the arranger supplies the absolute length, because "how big is a leaf" is a function
    /// of the plant's size, not of its species alone.
    public var aspect: Double

    /// Cross-blade taco cup — the margin lifts toward the presented face in proportion to `u`, so
    /// the midrib sits in a valley and the blade catches a shading gradient instead of reading as
    /// one flat facet.
    public var fold: Double

    /// Longitudinal gravity droop: the midrib bows away from the presented face as `v²`. A big
    /// soft leaf droops; a stiff succulent paddle or a small waxy citrus blade barely does.
    public var curl: Double

    /// Catmull-Rom subdivisions applied to the authored margin. Higher rounds the outline more, at
    /// `subdivisions ×` the triangles — so it is the tri-budget lever for any plant that multiplies
    /// blades (a fern's leaflets, a whole canopy).
    ///
    /// **1 means "leave the authored controls alone"**, which is the byte-identity path the yard
    /// tree's pinned triangle counts depend on. Anything that renders large and close wants 2–3.
    public var subdivisions: Int

    /// Detail tier for the outline — only lobed silhouettes carry a reduced variant.
    public var tier: LeafSilhouette.Tier

    public init(silhouette: LeafSilhouette,
                aspect: Double,
                fold: Double = LeafConstructor.defaultFold,
                curl: Double = 0,
                subdivisions: Int = 1,
                tier: LeafSilhouette.Tier = .hero) {
        self.silhouette = silhouette
        self.aspect = aspect
        self.fold = fold
        self.curl = curl
        self.subdivisions = subdivisions
        self.tier = tier
    }

    /// The resolved half-margin this blade emits with.
    public var margin: [LeafSilhouette.Control] {
        silhouette.margin(tier: tier, subdivisions: subdivisions)
    }

    /// Emit one blade of this species at `position`, `length` metres long.
    ///
    /// The width follows from `aspect`, so a caller decides only how big the plant's leaves are —
    /// never how broad this species' leaf is relative to its length, which is the species' business.
    public func emit<Sink: LeafCardSink>(into sink: inout Sink,
                                         position: SIMD3<Sink.Scalar>,
                                         xAxis: SIMD3<Sink.Scalar>,
                                         yAxis: SIMD3<Sink.Scalar>,
                                         bentNormal: SIMD3<Sink.Scalar>,
                                         length: Sink.Scalar,
                                         winding: LeafConstructor.Winding) {
        let placement = LeafConstructor.Placement(
            position: position, xAxis: xAxis, yAxis: yAxis, bentNormal: bentNormal,
            width: length * Sink.Scalar(aspect), height: length)
        LeafConstructor.emitBlade(into: &sink,
                                  placement: placement,
                                  margin: margin,
                                  fold: Sink.Scalar(fold),
                                  curl: Sink.Scalar(curl),
                                  winding: winding)
    }
}

/// Where a species puts its blades.
///
/// **This enum is a specification, not yet an implementation** — deliberately. Two tuned, gated
/// arrangers ship today (`PottedPlantMesh.foliageMesh` for pot/garden plants, and
/// `ForestTreeGeometry.emitLeaves` + `fillCrownVolume` for the canopy), and the honest way to get
/// to one is to lift THOSE, preserving their output, rather than to author a third here and leave
/// three. The cases below name the topologies that lift has to cover, so the work has a target and
/// a new species has somewhere to say which habit it has.
///
/// Once the lift lands, a species names its arrangement and the shared arranger realizes it — which
/// is the point at which "adding a species" really is describing a silhouette plus a growth habit.
public enum LeafArrangement: Sendable, Hashable {
    /// A basal whorl with no visible stem — jade, echeveria, agave, most succulents.
    case rosette
    /// Blades spiralling up a single stem at the golden angle — most upright houseplants.
    case stemSpiral
    /// A rachis carrying paired leaflets — ferns, palms, and every pinnate leaf.
    case compoundFrond
    /// Blades on the twigs of a recursive branch armature — a tree.
    case twigPhyllotaxis
    /// Radial spines from a fleshy body rather than blades at all — cactus.
    case areole
}

// MARK: - The shipped species' blades
//
// A catalog of constants, not a switch. Values carried over from the two historical descriptions
// so the shipped plants are unchanged; new species are added here rather than in a consumer.

extension BladeDescription {

    // Houseplants (from `PottedPlantMesh.foliagePlan`).

    /// One huge simple glossy sheet on a short petiole.
    public static let fiddleLeafFig = BladeDescription(
        silhouette: .fiddleLeafFig, aspect: 0.62, curl: 0.30, subdivisions: 3)

    /// Broad fenestrated tropical blade, heavy enough to droop.
    public static let monstera = BladeDescription(
        silhouette: .monstera, aspect: 0.92, curl: 0.34, subdivisions: 3)

    /// Stiff upright sword — a snake plant barely droops at all.
    public static let snakePlant = BladeDescription(
        silhouette: .strapBlade, aspect: 0.16, fold: 0.34, curl: 0.05, subdivisions: 2)

    /// Fat waxy paddle; rigid, so almost no curl.
    public static let jade = BladeDescription(
        silhouette: .succulentPad, aspect: 0.74, fold: 0.38, curl: 0.04, subdivisions: 3)

    /// One pinnule of a fern frond — small and numerous, so the cheapest rounding that still reads.
    public static let fernLeaflet = BladeDescription(
        silhouette: .fernLeaflet, aspect: 0.42, fold: 0.26, curl: 0.12, subdivisions: 2)

    /// A bloom petal, not a leaf: rounder margin, shallower cup.
    public static let petal = BladeDescription(
        silhouette: .petal, aspect: 0.58, fold: 0.24, curl: 0.10, subdivisions: 2)

    // Trees (from `ForestTreeGeometry.TreeLook`). `subdivisions: 1` is load-bearing — the canopy
    // multiplies every blade by thousands, and the baked triangle counts are pinned.

    public static let oakLeaf = BladeDescription(
        silhouette: .oak, aspect: 0.66, curl: 0.22, subdivisions: 1)

    public static let birchLeaf = BladeDescription(
        silhouette: .birch, aspect: 0.72, curl: 0.22, subdivisions: 1)

    public static let mapleLeaf = BladeDescription(
        silhouette: .maple, aspect: 0.94, curl: 0.22, subdivisions: 1)

    public static let citrusLeaf = BladeDescription(
        silhouette: .citrus, aspect: 0.54, curl: 0.26, subdivisions: 1)
}
