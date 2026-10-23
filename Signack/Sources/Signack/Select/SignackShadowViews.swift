import simd

/// Per-light shadow culling (Daydream DH-1465) — the CPU REFERENCE of the shadow-view half of the
/// GPU cut pass.
///
/// S2 culls a cut to the CAMERA only for the G-buffer and lets every shadow pass draw the whole LOD
/// cut, because a caster off screen still throws a shadow on screen. That is correct and wasteful:
/// a sun cascade slice, a spot cone or a point-light cube face only ever rasterises what is inside
/// ITS OWN frustum. So each shadow view gets its own list: the clusters of the (camera-chosen) LOD
/// cut whose wind-padded bounds can touch that view's frustum. Culling never changes WHICH level is
/// chosen — every shadow view draws a subset of the same crack-free cut — so it cannot open a crack,
/// and a cluster that can reach a view's frustum is drawn in it, so it cannot drop a shadow.
///
/// The shadow passes clip depth (no depth clamp), so all six planes of a view are real raster
/// bounds; the test is the same padded-sphere test the camera frustum uses.
public enum SignackShadowViews {
    /// The most shadow views one cut carries (the GPU's view mask is one byte).
    public static let maxViews = 8

    /// For each of `views` (at most `maxViews`, object-space frustums), the clusters of `view`'s LOD
    /// cut that can land in it, ascending. Padding is `view.cullPadding` (the vertex stage's reach).
    public static func select(_ h: SignackHierarchy, view: SignackView,
                              shadowViews views: [SignackFrustum]) -> [[UInt32]] {
        let cut = SignackSelection.select(h, view: view)
        return views.prefix(maxViews).map { f in
            cut.filter { f.intersects(SignackSelection.padded(h.clusters[Int($0)].bounds, view)) }
        }
    }
}
