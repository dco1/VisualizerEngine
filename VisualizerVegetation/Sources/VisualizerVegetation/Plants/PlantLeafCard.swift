import Foundation
import simd
import VisualizerMaterials

/// Potted-plant leaf blade — a THIN adapter over the shared engine constructor.
///
/// This used to be a hand-ported DUPLICATE of `ForestTreeGeometry.emitLeafCard`'s HERO silhouette
/// path (species margin outline + taco cup + strip stitch). It now delegates to
/// `VisualizerVegetation` — `LeafSilhouette` for the outline, `LeafConstructor.emitBlade` for the
/// stitch — so the potted-plant leaves, the vase-of-flowers petals AND the yard-tree bake all
/// derive from ONE source, in the shared engine submodule where the sibling app can reach it too
/// (DH-0651; CLAUDE.md [[feedback-single-source-of-truth]]).
///
/// The public surface (`Silhouette`, `emit`) is preserved so `PottedPlantMesh` is unchanged.
public enum PlantLeafCard {

    /// The broadleaf silhouette family for potted plants — each case NAMES one engine
    /// `LeafSilhouette` constant. (`petal` is not offered here; petals go through `PlantPetalCard`.)
    public enum Silhouette {
        case oak      // pinnately lobed obovate
        case birch    // doubly-serrated triangular-ovate
        case maple    // palmate-lobed
        case blade    // plain tapered strap (snake-plant sword blade)
        case fiddleLeafFig  // one huge simple entire violin/obovate blade
        case monstera       // broad blade with deep rounded marginal splits (fenestration read)
        case succulentPad   // short, fat, rounded fleshy paddle
        case fernLeaflet    // a small lanceolate pinnule of a compound fern frond

        public var canonical: LeafSilhouette {
            switch self {
            case .oak:   return .oak
            case .birch: return .birch
            case .maple: return .maple
            case .blade: return .strapBlade
            case .fiddleLeafFig: return .fiddleLeafFig
            case .monstera:      return .monstera
            case .succulentPad:  return .succulentPad
            case .fernLeaflet:   return .fernLeaflet
            }
        }
    }

    /// The half-margin `(v, u)` outline for a silhouette (the potted-plant HERO tier). Kept as a
    /// pass-through to the canonical table so any existing caller/test reads the single source.
    public static func margin(_ s: Silhouette) -> [(v: Double, u: Double)] {
        s.canonical.controls(tier: .hero).map { (v: $0.v, u: $0.u) }
    }

    /// Emit ONE leaf blade — delegates to `LeafConstructor.emitBlade` with the HERO silhouette, a
    /// per-style cross-blade taco `fold` (default the Forest HERO 0.30), and a per-style
    /// longitudinal gravity-`curl` (default 0, which is byte-identical to the former flat port). A
    /// houseplant leaf droops under its own weight; the curl bows the blade into a solid 3D arch so
    /// it reads beside the furniture, not as a flat cutout (DH-0488). `fold` is the ONLY geometric
    /// stand-in for a visible midrib this shared adapter offers — the margins lift toward
    /// `bentNormal` in proportion to how far across the blade they sit, so the spine reads as a
    /// shading valley even though every leaf still carries one flat stamped colour (the render
    /// bridge overwrites per-vertex colour with one green per style — DH-0645). A big simple-bladed
    /// species (fiddle-leaf fig) wants a deeper fold than a thin blade (snake plant) to read at all.
    public static func emit(_ mesh: inout Mesh3,
                     pos: Vec3, xAxis: Vec3, yAxis: Vec3, normal bentNormal: Vec3,
                     w: Double, h: Double, silhouette: Silhouette, curl: Double = 0,
                     subdivisions: Int = 1, fold: Double = 0.30,
                     waveAmplitude: Double = 0, wavePhase: Double = 0) {
        LeafConstructor.emitBlade(
            into: &mesh,
            placement: .init(position: pos, xAxis: xAxis, yAxis: yAxis,
                             bentNormal: bentNormal, width: w, height: h),
            silhouette: silhouette.canonical,
            tier: .hero,
            subdivisions: subdivisions,
            fold: fold,
            curl: curl,
            winding: .doubleSidedShell,
            waveAmplitude: waveAmplitude, wavePhase: wavePhase)
    }
}
