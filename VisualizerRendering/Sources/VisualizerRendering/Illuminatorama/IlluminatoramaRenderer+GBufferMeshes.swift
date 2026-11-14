import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    public func setMesh(_ kind: MeshKind, _ mesh: IlluminatoramaMesh) {
        // DH-1668 — a different mesh under a kind already in the table changes every instance's
        // shadow WITHOUT changing any instance ref, which is all the shadow reuse compares. The
        // old geometry's extent is gone with it, so the change cannot be bounded: re-render all.
        if let old = meshes[kind], old !== mesh { markCasterChangeUnbounded("mesh replaced under \(kind)") }
        meshes[kind] = mesh
    }

    // ── Phase 4.8 — GPU-direct mesh registration ───────────────────────
    //
    // Compute-fed callers (`DynamicMesh`, `BarkRenderer`, MLS-MPM solvers)
    // hand the renderer a ready-made `IlluminatoramaMesh` whose vertex /
    // index buffers they own. The renderer stores it in `meshes` under a
    // freshly-generated `MeshKind.custom(UUID)` so each registration is
    // unique; the returned `IlluminatoramaMeshHandle` carries the kind
    // back to the host, who attaches it to the relevant `SCNGeometry` via
    // `SCNGeometry.illuminatoramaMeshHandle`. The scene extractor checks
    // the associated object before falling through to the CPU-readback
    // path, so a single attach is all the wiring needed per geometry.

    @discardableResult
    /// Object-space AABB of a registered mesh, or `nil` if it cannot be read.
    /// Diagnostic path — see `IlluminatoramaMesh.localBounds()`.
    public func meshLocalBounds(_ kind: MeshKind) -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        meshes[kind]?.localBounds()
    }

    public func registerMesh(_ mesh: IlluminatoramaMesh) -> IlluminatoramaMeshHandle {
        let kind = MeshKind.custom("gpuMesh#\(UUID().uuidString)")
        meshes[kind] = mesh
        return IlluminatoramaMeshHandle(kind: kind, mesh: mesh, renderer: self)
    }

    /// The registered mesh behind a `MeshKind`, or nil if nothing is registered under it.
    ///
    /// Read-only lookup into the mesh table, for a host that needs the geometry it is DRAWING
    /// rather than a copy of what it once submitted — exact cursor picking (ray vs the real
    /// triangles, via `IlluminatoramaMesh.objectTriangleSoup()`), a measurement probe, a
    /// geometry audit. The alternative is for every host to shadow the table with its own
    /// `[MeshKind: IlluminatoramaMesh]`, which is a second source of truth for the same fact and
    /// goes stale exactly when a mesh is re-registered.
    ///
    /// The returned object is the renderer's own entry, not a copy: treat its buffers as
    /// read-only, and note a `.private`-storage mesh has no CPU-visible vertices at all
    /// (`objectTriangleSoup()` is empty for it).
    public func registeredMesh(_ kind: MeshKind) -> IlluminatoramaMesh? { meshes[kind] }

    /// How many meshes have left the table, ever — monotonic. A host that keeps instance refs across
    /// frames (Daydream's persistent element instances, DH-1675) re-checks they still name a mesh
    /// only when this moved, instead of hashing every kind every frame. Kinds are never reused
    /// (`registerMesh` mints a fresh UUID kind), so an unchanged count proves every kind it saw is live.
    public var meshTableRemovals: Int { kMisc.meshTableRemovals }

    /// Called by `IlluminatoramaMeshHandle.deinit` to evict the entry
    /// when the host drops its last strong reference. Internal because
    /// hosts should not call this directly — letting the handle's
    /// lifetime drive cleanup avoids accidental dangling MeshKind keys
    /// while an SCNNode still references the geometry.
    internal func removeMesh(_ kind: MeshKind) {
        // DH-1668 — a host that swaps a moved piece's mesh (Daydream's structural load drops the
        // old handle before the next frame renders) still needs the OLD geometry's bounds to clear
        // its shadow from the maps that hold it. Keep the departing mesh until the next stability
        // update has diffed last frame's instances against this frame's — only a mesh last frame
        // actually drew (`meshGroupRange` is that upload's grouping), so loads that never render
        // cannot pile meshes up here.
        if let gone = meshes.removeValue(forKey: kind) {
            kMisc.meshTableRemovals &+= 1
            if gbs.meshGroupRange[kind] != nil { shd.departedMeshes[kind] = gone }
        }
        fxs.gpuRepackTasks.removeAll { $0.kind == kind }
    }
}
