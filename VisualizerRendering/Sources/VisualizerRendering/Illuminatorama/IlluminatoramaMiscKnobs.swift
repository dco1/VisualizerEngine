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
    var meshes: [MeshKind: IlluminatoramaMesh] = [:]
}
