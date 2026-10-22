import simd

/// LOD selection — the CPU REFERENCE of the pass S2 runs on the GPU.
///
/// Data-parallel by construction: each cluster decides alone, from its own record and the view, in
/// a handful of flops — no traversal, no pointer chasing, no state from other clusters. A compute
/// shader runs `isInCut` one thread per cluster and compacts the survivors; this file does the same
/// with a loop.
public enum SignackSelection {

    /// Is `cluster` part of the crack-free cut for `view`? Its own error is acceptable and its
    /// parent group's is not. (See `SignackCluster` for why that is crack-free.)
    @inlinable public static func isInCut(_ c: SignackCluster, _ view: SignackView) -> Bool {
        view.projectedError(c.lodSphere, c.lodError) <= view.threshold
            && view.projectedError(c.parentSphere, c.parentError) > view.threshold
    }

    /// The cut for `view`, as ascending cluster ids. With `cull`, clusters outside `view.frustum`
    /// are dropped too (culling never changes WHICH level is chosen, so it cannot open a crack).
    public static func select(_ h: SignackHierarchy, view: SignackView, cull: Bool = false) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(h.clusters.count / 2)
        let frustum = cull ? view.frustum : nil
        for (i, c) in h.clusters.enumerated() where isInCut(c, view) {
            if let f = frustum, !f.intersects(c.bounds) { continue }
            out.append(UInt32(i))
        }
        return out
    }

    /// The view-independent cut keeping every cluster whose error is ≤ `tolerance` object units —
    /// fixed for the life of the bake (a ray tracer's proxy).
    public static func cut(_ h: SignackHierarchy, tolerance: Float) -> [UInt32] {
        select(h, view: .objectSpace(tolerance: tolerance))
    }

    /// Frustum-cull an existing cut.
    public static func cull(_ h: SignackHierarchy, _ cut: [UInt32], frustum: SignackFrustum) -> [UInt32] {
        cut.filter { frustum.intersects(h.clusters[Int($0)].bounds) }
    }
}
