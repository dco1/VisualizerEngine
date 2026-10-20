import simd
import Foundation

/// Bakes procedural materials into tileable `MaterialChannels`. This is the
/// composable engine of MATERIALS_AND_TEXTURES §7 expressed on the CPU: each generator
/// is a small graph of `Noise` nodes → height → derived normal/curvature → roughness +
/// albedo. The terrazzo here is the §1 hero — the Roominate material that "turned out
/// well", generalized: seeded aggregate-chip scatter (Voronoi), per-chip palette +
/// roughness variation, a slightly-rougher cement binder, and a polish clearcoat.
public enum MaterialGenerator {

    /// Default bake resolution for every procedural material tile. 512 matches the
    /// engine atlas's slice size EXACTLY — the atlas letterboxes whatever it's given
    /// into a 512² slice, so a 256 bake was silently UPSCALED 2× on upload and every
    /// material rendered pre-blurred (Danny's "materials look like bitmaps"; the old
    /// hex anti-tiling dithered the blur, and turning it off for coherent wood/tile
    /// exposed it). Keep this in lockstep with the renderer's `atlasSliceSize`.
    public static let bakeSize = 512

    /// Sub-cell phase offset (in UV) applied to the aggregate-stone Voronoi samplers
    /// (granite, pebble) so the tile boundary falls at a grain/stone CENTRE rather than
    /// on the Voronoi cell lattice line. Both stones are already perfectly toroidal, but
    /// a lattice-aligned tile edge is a grain-boundary-dense column, so TextureAudit's
    /// seam ratio (edge diff ÷ mean interior diff) reads high even with no visible seam.
    /// 26 cells span the tile → half a cell is `0.5 / 26`; sampling there makes the wrap
    /// column representative of the interior (seam ≈ 1). Value is arbitrary within a cell;
    /// half-cell maximises distance from every lattice line.
    static let grainPhase = 0.5 / 26.0
}
