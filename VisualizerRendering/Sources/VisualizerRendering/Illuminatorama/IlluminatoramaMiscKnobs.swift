import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - MiscKnobs

/// Stored state for the renderer's `misc` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `MiscKnobs` (`kMisc`) and forwards each name to it from
/// `IlluminatoramaRenderer+MiscKnobs.swift`; callers see no difference.
@MainActor
final class MiscKnobs {
    typealias MeshKind = IlluminatoramaRenderer.MeshKind

    /// When true, the scene extractor bakes a world-space triangle soup from
    /// the scene it walks and feeds it to `setRTGeometry` (dirty-rebuilt), so
    /// RT GI + soft shadows + reflections work on ANY extracted overlay scene —
    /// not just the hand-authored room that calls `setRTGeometry` directly.
    /// The extractor also points the RT sun at the scene's extracted key light.
    /// (Surface cache stays opt-in/manual — it needs per-card UVs the soup
    /// can't synthesise for arbitrary meshes.)
    var buildRTFromExtractedScene: Bool = false
    var meshes: [MeshKind: IlluminatoramaMesh] = [:] { didSet { meshTableVersion &+= 1 } }
    /// Bumped on every change to `meshes` (DH-1684: the GPU shadow draw table keys on it).
    var meshTableVersion = 0
    /// The same counter under the name the cull-bounds and draw-group caches read: what's derived from the table —
    /// each draw group's mesh (`meshGroupMeshes`), the cull bounds — knows when to look again, even while the draw
    /// list holds still and its upload is skipped.
    var meshesVersion: Int { meshTableVersion }
    /// How many meshes have left the table, ever (Daydream DH-1675).
    var meshTableRemovals = 0
}
