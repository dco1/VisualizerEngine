import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Whether the surface-cache update should run this frame.
    var surfCacheActive: Bool {
        surfaceCacheEnabled && rtEnabled && rtSupported
            && scs.surfCachePipeline != nil && scs.surfCardCount > 0 && rts.rtAccel != nil
    }

    /// Read a deforming (DynamicMesh / GPU-fed) mesh's CURRENT object-space
    /// triangle soup directly from the shared buffers the GPU traces: live
    /// positions from the repack task's `packed_float3` source buffer (the host
    /// or a kernel rewrites it every frame), indices from the mesh's combined
    /// index buffer. `IlluminatoramaMesh.objectTriangleSoup()` can't do this — the
    /// interleaved vertex buffer is `.private`, so it returns empty. Returns nil
    /// for a non-deforming kind, a private source buffer, or empty topology.
    func liveDeformingObjectSoup(kind: MeshKind)
        -> (positions: [SIMD3<Float>], indices: [UInt32])?
    {
        guard let mesh = meshes[kind],
              let task = fxs.gpuRepackTasks.first(where: { $0.kind == kind }),
              task.positionBuffer.storageMode == .shared,
              mesh.indexBuffer.storageMode == .shared else { return nil }
        let n = task.vertexCount
        guard n > 0, mesh.indexCount >= 3 else { return nil }
        // Source positions are packed_float3 (stride 12) — the layout the repack
        // kernel + BLAS read. Reading them as SIMD3<Float> (stride 16) would shear
        // every vertex past the first (the SIMD3 stride trap).
        let pbase = task.positionBuffer.contents().bindMemory(to: Float.self, capacity: n * 3)
        var positions = [SIMD3<Float>](); positions.reserveCapacity(n)
        for v in 0..<n { positions.append(SIMD3<Float>(pbase[3*v], pbase[3*v+1], pbase[3*v+2])) }
        let ibase = mesh.indexBuffer.contents()
        var indices = [UInt32](); indices.reserveCapacity(mesh.indexCount)
        if mesh.indexType == .uint16 {
            for i in 0..<mesh.indexCount { indices.append(UInt32(ibase.load(fromByteOffset: i * 2, as: UInt16.self))) }
        } else {
            for i in 0..<mesh.indexCount { indices.append(ibase.load(fromByteOffset: i * 4, as: UInt32.self)) }
        }
        return (positions, indices)
    }

    /// Re-frame dynamic surface cards in place each frame, BEFORE
    /// `encodeSurfaceCacheUpdate`, so the update kernel re-lights moved/deformed
    /// surfaces at their CURRENT pose. Two dynamic sources, one shared mechanism
    /// (write the card's plane + flag it dirty → kernel uses α=1 to discard stale
    /// history while stationary cards keep their accumulated multi-bounce):
    ///
    ///  • **Rigid incremental invalidation** (`surfaceCacheIncremental`, default
    ///    on): an instance whose model matrix changed re-frames from its STATIC
    ///    object soup × the new transform. See
    ///    docs/illuminatorama/surface-cache-incremental-invalidation.md.
    ///  • **Phase C — deforming meshes** (DynamicMesh / GPU-fed): a mesh whose
    ///    *shape* changes in place (the transform may never move, so the rigid
    ///    diff misses it). Re-frames from this frame's LIVE object soup ×
    ///    transform, every frame. This is what makes a rippling sheet / sloshing
    ///    fluid / waving flag both CATCH the room's bounced colour (its cards sit
    ///    on the real surface, sampling neighbours' caches) and CAST it back
    ///    (its now-correctly-placed cards re-light and are sampled by others'
    ///    GI/reflection rays — and, on the TLAS update path, with current-pose
    ///    occlusion). On by default whenever a deforming mesh drives the cache;
    ///    `VIZ_ILLUMI_NO_DEFORM_CARDS=1` freezes it at build pose for A/B.
    ///
    /// PERF (Phase C → D): the deforming re-frame was a per-frame CPU readback +
    /// transform of the live soup; Phase D moved it to
    /// `illumi_surfcache_reframe_deform` (a compute kernel writing
    /// `surfCardBuffer` straight from the solver's live position buffer). The
    /// CPU loop below remains only as the `VIZ_SURFCACHE_GPU_DIFF=0` fallback.
    ///
    /// `surfCardBuffer` is written in place (not ping-ponged). On the default GPU lane that
    /// is a kernel on this frame's command buffer, ordered by hazard tracking; on the CPU
    /// fallback it is a host write that first waits for the in-flight frame
    /// (`awaitInFlightFrameBeforeCPUCacheWrite`, DH-0537 #1) — the pool is 2.
    ///
    /// **#60 item 1 + Phase D — GPU-side transform diff.** Both dynamic sources
    /// now run as compute kernels on the frame's command buffer: RIGID movers
    /// via `illumi_surfcache_reframe_tri` / `_chart` (each thread diffs its
    /// owner instance's current-vs-previous model matrix inline and re-frames
    /// its card (half) in place), DEFORMING meshes via
    /// `illumi_surfcache_reframe_deform` (one dispatch per deforming instance,
    /// reading the solver's live position buffer — no CPU readback). All write
    /// the same `surfCardBuffer` / dirty flags the update kernel consumes later
    /// in the SAME command buffer (hazard tracking orders them). The CPU loops
    /// below remain solely as the `VIZ_SURFCACHE_GPU_DIFF=0` kill-switch / A/B
    /// reference (and the safety fallback when a map or live buffer is
    /// unresolvable mid-topology).
    func applyIncrementalSurfaceCacheUpdate(_ cb: MTLCommandBuffer) {
        guard surfCacheActive, scs.surfIncrementalReady,
              let dirtyBuf = scs.surfCardDirtyBuffer, let cardBuf = scs.surfCardBuffer else { return }
        // Chart path (`VIZ_SURFCACHE_CHARTS=1`): re-frame each moved rigid instance's
        // cards by a single affine transform of their baked object-space frame.
        if let chart = scs.surfChartIncremental {
            if surfaceCacheIncremental, !Self.surfCacheGPUDiffDisabled,
               let pipe = scs.surfReframeChartPipeline,
               let ownerBuf = scs.surfGPUDiffChartOwnerBuffer,
               let frameBuf = scs.surfGPUDiffChartFrameBuffer,
               let statsBuf = surfGPUDiffStatsBuffer,
               chart.instCardLo.count == instances.count {
                encodeGPUDiffReframe(cb, pipeline: pipe, threadCount: scs.surfCardCount,
                                     dirtyBuf: dirtyBuf, cardBuf: cardBuf,
                                     ownerBuf: ownerBuf, statsBuf: statsBuf,
                                     chartFrameBuf: frameBuf, tag: "[chart gpu]")
                return
            }
            applyIncrementalChartSurfaceCacheUpdate(chart, dirtyBuf: dirtyBuf, cardBuf: cardBuf)
            return
        }
        // Per-triangle path: GPU kernels (#60 item 1 + Phase D). The rigid
        // kernel owns every non-deforming card's diff + re-frame + dirty flag;
        // one deform dispatch per deforming instance re-frames its cards from
        // the LIVE position buffer (no readback). Falls back to the CPU loop
        // below when the kill-switch is set or any map / pipeline / live
        // buffer is missing.
        if !Self.surfCacheGPUDiffDisabled,
           encodeGPUDiffPerTriPath(cb, dirtyBuf: dirtyBuf, cardBuf: cardBuf) {
            return
        }
        guard scs.surfSoupBaseCPU.count == instances.count,        // topology matches the baked maps
              !scs.surfTriCardCPU.isEmpty else { return }

        // Which kinds deform in place this frame? (Phase C re-frames them always.)
        let deformingKinds: Set<MeshKind> =
            (scs.surfHasDeformingCards && !scs.noDeformCardsOverride) ? Set(fxs.gpuRepackTasks.map(\.kind)) : []
        let rigidIncremental = surfaceCacheIncremental
        // Nothing dynamic → leave every card fully accumulating (cheapest path).
        guard rigidIncremental || !deformingKinds.isEmpty else { return }

        // Snapshot each deforming kind's live object soup ONCE (shared by all its
        // instances). Indices are constant topology; positions are this frame's.
        var liveSoupByKind: [MeshKind: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
        for kind in deformingKinds {
            if let soup = liveDeformingObjectSoup(kind: kind) { liveSoupByKind[kind] = soup }
        }

        let total = instances.count
        let curPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let prevPtr = previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)

        // CPU fallback: it rewrites the card + dirty buffers the in-flight frame may be reading.
        awaitInFlightFrameBeforeCPUCacheWrite()
        // Clear last frame's dirty flags; cards that stop moving revert to EMA.
        memset(dirtyBuf.contents(), 0, dirtyBuf.length)
        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: scs.surfCardCount)
        let cards = cardBuf.contents().bindMemory(to: SurfCard.self, capacity: scs.surfCardCount)
        let soupTriTotal = scs.surfTriCardCPU.count

        func moved(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
            let e: Float = 1e-5
            for c in 0..<4 {
                let d = a[c] - b[c]
                if abs(d.x) > e || abs(d.y) > e || abs(d.z) > e || abs(d.w) > e { return true }
            }
            return false
        }

        var movedInstances = 0
        var deformedInstances = 0
        var dirtiedCards = 0
        for i in 0..<total {
            let kind = scs.surfInstanceKind[i]
            let isDeforming = deformingKinds.contains(kind)
            let model = curPtr[i].modelMatrix
            // Re-frame when: a deforming instance (its verts changed — always), OR
            // a rigid instance whose transform moved (incremental invalidation).
            let soup: (positions: [SIMD3<Float>], indices: [UInt32])?
            if isDeforming {
                soup = liveSoupByKind[kind]
                deformedInstances += 1
            } else {
                guard rigidIncremental, moved(model, prevPtr[i].modelMatrix) else { continue }
                soup = scs.surfObjectSoupByKind[kind]
                movedInstances += 1
            }
            guard let soup, !soup.indices.isEmpty else { continue }
            let firstTri = Int(scs.surfSoupBaseCPU[i])
            let nextTri = (i + 1 < total) ? Int(scs.surfSoupBaseCPU[i + 1]) : soupTriTotal
            let localCount = min(nextTri - firstTri, soup.indices.count / 3)
            for j in 0..<max(0, localCount) {
                let gt = firstTri + j
                guard gt < soupTriTotal else { break }
                let i0 = Int(soup.indices[3*j]), i1 = Int(soup.indices[3*j+1]), i2 = Int(soup.indices[3*j+2])
                guard i0 < soup.positions.count, i1 < soup.positions.count, i2 < soup.positions.count else { continue }
                func world(_ p: SIMD3<Float>) -> SIMD3<Float> { let w = model * SIMD4<Float>(p, 1); return SIMD3(w.x, w.y, w.z) }
                let v0 = world(soup.positions[i0]), v1 = world(soup.positions[i1]), v2 = world(soup.positions[i2])
                var n = simd_cross(v1 - v0, v2 - v0)
                let len = simd_length(n); n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
                let card = Int(scs.surfTriCardCPU[gt])
                guard card < scs.surfCardCount else { continue }
                // Re-frame only this triangle's half of its (possibly packed) card;
                // preserve the packed flag (normal.w) + albedo/emission.
                if scs.surfTriIsFrameB[gt] {
                    cards[card].originB = SIMD4(v0, 0); cards[card].uAxisB = SIMD4(v1 - v0, 0)
                    cards[card].vAxisB = SIMD4(v2 - v0, 0); cards[card].normalB = SIMD4(n, cards[card].normalB.w)
                } else {
                    cards[card].origin = SIMD4(v0, 0); cards[card].uAxis = SIMD4(v1 - v0, 0)
                    cards[card].vAxis = SIMD4(v2 - v0, 0); cards[card].normal = SIMD4(n, cards[card].normal.w)
                }
                if dirty[card] == 0 { dirtiedCards += 1 }
                dirty[card] = 1
            }
        }
        // Diagnostic (env-gated, sandbox-safe sidecar): proves the dynamic path
        // fired this frame and how much it touched. No-op without
        // VIZ_ILLUMI_SURFCACHE_STATS_PATH.
        if movedInstances > 0 || deformedInstances > 0 {
            Self.recordSurfCacheStats("dynamic: moved=\(movedInstances) deformed=\(deformedInstances) dirtiedCards=\(dirtiedCards) of \(scs.surfCardCount)")
        }
    }

    /// Chart-path counterpart of `applyIncrementalSurfaceCacheUpdate` (#60 task 1).
    ///
    /// A coplanar chart card spans many triangles under ONE bbox-derived frame, so
    /// the per-triangle re-frame would clobber its parameterisation. Instead we use
    /// the card frame's **affine covariance**: for a rigid/affine model matrix `M`,
    /// the world frame is exactly `M · objectFrame` — `origin` as a point, `uAxis`
    /// / `vAxis` as vectors — and the fixed per-triangle UVs are preserved (verts
    /// and frame transform together). So each moved instance's cards re-frame with
    /// one matrix multiply per axis, no per-triangle work, no UV touch. Cards owned
    /// by a moved instance get α=1 (dirty) so they re-light from the new pose;
    /// every stationary card keeps its converged radiance.
    private func applyIncrementalChartSurfaceCacheUpdate(_ chart: ChartIncremental,
                                                         dirtyBuf: MTLBuffer, cardBuf: MTLBuffer) {
        // Charts are rigid-mover only — the deforming per-card path doesn't apply.
        guard surfaceCacheIncremental else { return }
        awaitInFlightFrameBeforeCPUCacheWrite()   // in-place card/dirty rewrite below
        let total = instances.count
        guard chart.instCardLo.count == total, chart.instCardHi.count == total,
              chart.objectFrame.count == scs.surfCardCount else { return }
        let curPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let prevPtr = previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)

        func moved(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
            let e: Float = 1e-5
            for c in 0..<4 {
                let d = a[c] - b[c]
                if abs(d.x) > e || abs(d.y) > e || abs(d.z) > e || abs(d.w) > e { return true }
            }
            return false
        }

        // Which instances moved this frame? (Decide before clearing dirty flags.)
        // Reuse the bake-time scratch buffer (no per-frame allocation); reset in
        // place. Fall back to a fresh buffer only if it's somehow mis-sized.
        if scs.surfChartMovedScratch.count != total {
            scs.surfChartMovedScratch = [Bool](repeating: false, count: total)
        } else {
            for i in 0..<total { scs.surfChartMovedScratch[i] = false }
        }
        var movedInstances = 0
        for i in 0..<total where moved(curPtr[i].modelMatrix, prevPtr[i].modelMatrix) {
            scs.surfChartMovedScratch[i] = true; movedInstances += 1
        }
        // Clear last frame's dirty flags so cards that stopped moving revert to EMA
        // (same reset semantics as the per-triangle path); skip the rest if static.
        memset(dirtyBuf.contents(), 0, dirtyBuf.length)
        guard movedInstances > 0 else { return }

        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: scs.surfCardCount)
        let cards = cardBuf.contents().bindMemory(to: SurfCard.self, capacity: scs.surfCardCount)
        var dirtiedCards = 0
        for i in 0..<total where scs.surfChartMovedScratch[i] {
            let lo = Int(chart.instCardLo[i]), hi = Int(chart.instCardHi[i])
            guard lo >= 0, hi > lo, hi <= scs.surfCardCount else { continue }
            let m = curPtr[i].modelMatrix
            for c in lo..<hi {
                let f = chart.objectFrame[c]
                let o = m * SIMD4<Float>(f.origin, 1)
                let u = m * SIMD4<Float>(f.uAxis, 0)
                let v = m * SIMD4<Float>(f.vAxis, 0)
                let nn = m * SIMD4<Float>(f.normal, 0)
                var n = SIMD3<Float>(nn.x, nn.y, nn.z)
                let nl = simd_length(n); n = nl > 1e-8 ? n / nl : SIMD3<Float>(0, 1, 0)
                cards[c].origin = SIMD4(o.x, o.y, o.z, 0)
                cards[c].uAxis  = SIMD4(u.x, u.y, u.z, 0)
                cards[c].vAxis  = SIMD4(v.x, v.y, v.z, 0)
                cards[c].normal = SIMD4(n, cards[c].normal.w)   // preserve packed flag (0 for charts)
                if dirty[c] == 0 { dirtiedCards += 1 }
                dirty[c] = 1
            }
        }
        Self.recordSurfCacheStats("dynamic: moved=\(movedInstances) deformed=0 dirtiedCards=\(dirtiedCards) of \(scs.surfCardCount) [chart]")
    }
}
