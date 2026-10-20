import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    public func setMesh(_ kind: MeshKind, _ mesh: IlluminatoramaMesh) {
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

    /// Called by `IlluminatoramaMeshHandle.deinit` to evict the entry
    /// when the host drops its last strong reference. Internal because
    /// hosts should not call this directly — letting the handle's
    /// lifetime drive cleanup avoids accidental dangling MeshKind keys
    /// while an SCNNode still references the geometry.
    internal func removeMesh(_ kind: MeshKind) {
        meshes.removeValue(forKey: kind)
        fxs.gpuRepackTasks.removeAll { $0.kind == kind }
    }
}
