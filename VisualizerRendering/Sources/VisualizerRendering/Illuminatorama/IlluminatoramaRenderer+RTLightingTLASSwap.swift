import Metal

// TLAS SWAP REBUILDS ARE NOT THRASH (Daydream DH-1668)
// ===================================================
//
// The rebuild-thrash guard turns ray tracing off after `rtRebuildThrashLimit` topology changes in
// a row, because a scene that regenerates heavy geometry every frame (Forest's leaf field) would
// otherwise build big BLASes every frame and hang the GPU. An EDIT also changes topology every
// frame — Daydream's structural load re-registers each merged wall group it moved, so the TLAS
// sees ~30 new mesh kinds per drag frame — and so a sustained edit tripped the guard: RT glass,
// reflections and traced shadows dropped to their fallbacks mid-gesture and came back only after
// the topology held for the recovery window.
//
// What makes a rebuild dangerous is the NEW BLAS work, not the change itself: every mesh the TLAS
// already holds keeps its cached BLAS (`rtMeshCache`, keyed by the mesh object). So a change is a
// SWAP — rebuilt every time, never counted toward the guard — when:
//   · a TLAS exists and the curve registry has not changed since it was built;
//   · the instance count is within max(16, n/16) of the built one (an edit, not a new scene);
//   · the triangles of every mesh that has no BLAS yet sum to at most `rtSwapNewTriangleBudget`.
// The rebuild itself is the ordinary `rebuildRTAccel` — it traces the moved geometry, never a stale
// BLAS. A/B lever: `tlasSwapRebuildEnabled` (env `VIZ_ILLUMI_NO_TLAS_SWAP=1`).
//
// …and only while swaps stay CHEAP. The guard also protects the frame rate, and a swap that costs
// more than `rtSwapRebuildBudgetMs` of CPU is, to the user, exactly the thrash it guards against:
// on 4000 Sunset a wall edit's swap rebuild measured ~20 ms (55 new BLASes, ~11 ms of it the build
// wait, ~5 ms the normal/UV concatenation), and exempting it held RT on at 105 ms a frame against
// 73 ms with the guard tripped. So the last swap-shaped rebuild's cost decides whether the next one
// is exempt: a small edit keeps RT live through the gesture, a heavy one trips the guard as before.
extension IlluminatoramaRenderer {

    /// The most not-yet-built BLAS triangles a rebuild may carry and still be a swap.
    static let rtSwapNewTriangleBudget = 262_144
    /// The most CPU ms the last swap-shaped rebuild may have cost for the next one to be exempt.
    static let rtSwapRebuildBudgetMs: Double = 8

    /// The CPU ms the last full TLAS rebuild took, BLAS builds and waits included. Diagnostic.
    public var rtLastTLASRebuildMs: Double { rts.rtLastRebuildMs }
    /// Where the last rebuild's CPU time went. Diagnostic.
    public var rtLastTLASRebuildPhases: String { rts.rtLastRebuildPhases }

    func rtTopologyChangeIsSwap() -> Bool {
        guard tlasSwapRebuildEnabled, rts.rtTLAS != nil, rts.rtTLASInstanceCount > 0,
              rts.rtCurveSyncVersion == rts.rtTLASBuiltCurveVersion else { return false }
        let glass = rtGlassEnabled ? flattenGlass() : []
        var now = instances.count + rts.rtCurveSets.count
        for g in glass { now += g.insts.count }
        let was = rts.rtTLASInstanceCount
        guard abs(now - was) <= max(16, was / 16) else { return false }
        var newTriangles = 0
        func charge(_ mesh: IlluminatoramaMesh) -> Bool {
            guard rts.rtMeshCache[ObjectIdentifier(mesh)] == nil else { return true }
            newTriangles += mesh.indexCount / 3
            return newTriangles <= Self.rtSwapNewTriangleBudget
        }
        for g in gbs.meshGroups {
            guard let mesh = meshes[g.kind], mesh.indexCount >= 3, !rtGroupIsRayExcluded(g) else { continue }
            guard charge(mesh) else { return false }
        }
        for g in glass {
            guard let mesh = meshes[g.kind], mesh.indexCount >= 3 else { continue }
            guard charge(mesh) else { return false }
        }
        return true
    }
}
