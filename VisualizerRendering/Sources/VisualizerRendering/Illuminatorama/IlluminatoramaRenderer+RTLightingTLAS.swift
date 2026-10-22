import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    // ── TLAS update (instanced RT for animated extracted scenes) ──────
    //
    // Called from `render()` after `uploadInstances()` (so `meshGroups` +
    // `currentInstanceBuffer` are this frame's, grouped). On a topology change
    // (new mesh set / instance count) does a full BLAS+TLAS rebuild; otherwise
    // refits the TLAS in place on the frame's command buffer — cheap, no
    // per-frame vertex transform, no geometry rebuild. `rtTLASActive` tells the
    // RT pass to trace the instance AS instead of the primitive-AS soup.

    func rtTopologyHash() -> Int {
        var h = instances.count
        for g in gbs.meshGroups { h = h &* 31 &+ g.kind.hashValue; h = h &* 31 &+ g.count }
        // Curve sets are TLAS topology too (#60 item 7). The registry version
        // covers identity changes (re-register ⇒ rebuild even at equal counts).
        h = h &* 31 &+ rts.rtCurveSyncVersion
        h = h &* 31 &+ rts.rtCurveSets.count
        for s in rts.rtCurveSets {
            h = h &* 31 &+ s.segmentIndices.count
            h = h &* 31 &+ s.controlPoints.count
        }
        // AAA glass (#60): glass count + per-group kind/count are TLAS topology too
        // (a changed glass count must trigger a rebuild, not a refit).
        if rtGlassEnabled {
            for (kind, insts) in flattenGlass() {
                h = h &* 31 &+ kind.hashValue
                h = h &* 31 &+ insts.count
            }
        }
        return h
    }

    /// ONE place that turns the RT hang-guard off, so the two trip sites can't
    /// drift in what state they leave behind. `recordTrip` reports whether this
    /// was a real transition, so the log fires once per trip rather than once
    /// per frame (both trips are re-evaluated every frame now, and a per-frame
    /// `warning` would be a log flood for the whole length of a drag).
    func tripRTGuard(_ trip: IlluminatoramaRTGuardTrip, _ detail: String) {
        rts.rtTLASActive = false
        guard rts.rtGuard.recordTrip(trip) else { return }
        Self.log.warning("Illuminatorama RT auto-disabled (\(trip.rawValue)) — \(detail). Falling back to non-RT.")
    }

    func updateRTAccel(_ cb: MTLCommandBuffer) {
        // P1c — surface-cache scenes now ride the TLAS too: the lighting pass
        // reads cards via `soupTriBase[instance_id] + primitive_id`, while the
        // cache-UPDATE pass keeps tracing the grouped soup `rtAccel` that
        // `rebuildRTAccel`→`buildGroupedSurfaceSoup` bakes in TLAS-instance order.
        // Two triggers for a live TLAS: the extracted-scene RT path (opaque
        // lighting), OR AAA glass opted in (#60) — a native glass scene with no
        // opaque instances (Floating Lenses) still needs the TLAS for the glass
        // pass to trace. Both ride the same build below.
        let glassGroups = rtGlassEnabled ? flattenGlass() : []
        let hasGlass = !glassGroups.isEmpty
        let extractedRT = buildRTFromExtractedScene && rtEnabled
            && !gbs.meshGroups.isEmpty && !instances.isEmpty
        // Third trigger (S3.4): RT reflections opted into the TLAS on their own. Rides
        // the glass-arm caps below (extractedRT stays false), so it can never trip the
        // strict live-RT pair on a real document — see `rtTLASForReflections`.
        let reflectionsRT = rtTLASForReflections && rtEnabled && rtReflectionsEnabled
            && !gbs.meshGroups.isEmpty && !instances.isEmpty
        // Fourth trigger (S4.1): ray-traced soft SUN shadows in the deferred kernel
        // want the TLAS even with reflections compositing OFF (`rtReflectionsEnabled`
        // false — shadows want the geometry, not the composite). Same shape as the
        // reflections trigger and the same glass-arm caps, for the same reason: the
        // strict live-RT pair auto-disables on a real document (S3.4's finding), and
        // this feature must not be able to regress glass.
        let sunShadowRT = rtSunSoftShadowsEnabled && rtTLASForReflections && rtEnabled
            && !gbs.meshGroups.isEmpty && !instances.isEmpty
        // Fifth trigger (Daydream DH-0911): ray-traced AO claims the TLAS in its own right.
        // `rtaoActive` requires a live TLAS, but nothing let RTAO ASK for one — so on a host's
        // settled canvas (no extracted scene, reflections and traced sun both off) RTAO ran only
        // when the scene happened to contain glass and the glass pass built a TLAS anyway. The
        // host already requests it (it sets `rtTLASForReflections` whenever it wants RTAO); this
        // is the engine honouring that request. Measured before: 0 of 32 settle frames traced AO
        // in a room with no glazing, 31 of 32 with it. Same shape and glass-arm caps as the
        // reflections and sun-shadow triggers above, for the same reason; gated on intensity so
        // a zero-strength AO never builds a TLAS for nothing.
        let aoRT = rtaoEnabled && rtaoIntensity > 0 && rtTLASForReflections && rtEnabled
            && !gbs.meshGroups.isEmpty && !instances.isEmpty
        guard rtTLASSupported, (extractedRT || hasGlass || reflectionsRT || sunShadowRT || aoRT) else {
            rts.rtTLASActive = false; return }
        // Curve primitives (#60 item 7): adopt registry changes BEFORE the
        // topology hash, so a registration/unregistration lands as a rebuild.
        syncCurveRegistry()
        // Hang-proof gate 1 (up-front size cap — the PRIMARY defence): a scene
        // this heavy would stall on its very first rebuild (which never returns
        // for minutes), so the thrash counter below could never catch it. Reject
        // BEFORE any build, based on a cheap cost estimate. Re-evaluated EVERY
        // frame (the O(1) caps below bound the triangle sum to a loop over at
        // most `groupCap` mesh groups — 128 on the live-RT arm, 512 on the
        // glass-only arm since S3.6), so a scene that
        // sheds geometry — a floor hidden, a city context switched off — gets RT
        // back without a scene re-attach.
        var glassInstCount = 0
        for (_, insts) in glassGroups { glassInstCount += insts.count }
        // A glass-only TLAS (glass opted in, full-scene RT lighting off) traces from
        // only the glass fragments, so it carries a much higher triangle ceiling than
        // the per-pixel RT-lighting path — an interior with a lawn otherwise trips the
        // strict cap and its windows silently drop to the flat fallback.
        // S4.4 — a still under `rtStillCapsRelaxed` takes the glass-arm ceilings even on the
        // extracted-RT path (see the flag's note); the live loop keeps the strict pair.
        let strictLive = extractedRT && !rtStillCapsRelaxed
        let triCap = strictLive ? Self.rtMaxTrianglesForLiveRT
                                : rtGlassOnlyTriangleCap
        // …and the same asymmetry on the mesh-GROUP axis. A furnished architectural
        // document carries 200+ distinct geometries (one group per wall edge / room floor /
        // millwork sub-part), so the live-RT 128 turned AAA glass off scene-wide on every
        // real document. See `rtMaxMeshGroupsForGlassOnlyRT` for the measured ladder.
        let groupCap = rtMeshGroupCapOverride
            ?? (strictLive ? Self.rtMaxMeshGroupsForLiveRT
                           : Self.rtMaxMeshGroupsForGlassOnlyRT)
        // O(1) caps first, triangle sum only if they pass. This ordering matters
        // now that the guard re-evaluates every frame instead of latching: a
        // scene that busts the mesh-group cap (thousands of groups) would
        // otherwise pay a full dictionary-lookup loop every frame purely to
        // re-confirm it is still too heavy.
        var overCap = instances.count + glassInstCount > Self.rtMaxInstancesForLiveRT
            || gbs.meshGroups.count > groupCap
        var estTriangles = -1                     // −1 = not computed (cap already busted)
        if !overCap {
            estTriangles = 0
            for g in gbs.meshGroups {
                if let m = meshes[g.kind], !rtGroupIsRayExcluded(g) { estTriangles += m.indexCount / 3 }
            }
            for (kind, _) in glassGroups {
                if let m = meshes[kind] { estTriangles += m.indexCount / 3 }
            }
            overCap = estTriangles > triCap
        }
        if overCap {
            let tris = estTriangles < 0 ? "n/a" : "~\(estTriangles)"
            tripRTGuard(.sceneTooHeavy, """
                scene too heavy for live-loop RT \
                (\(instances.count) instances, \(gbs.meshGroups.count) mesh groups, \
                \(tris) tris; caps \
                \(Self.rtMaxInstancesForLiveRT)/\(groupCap)/\(triCap) — \
                \(strictLive ? "live-RT" : (extractedRT ? "still-RT" : "glass-only")) path)
                """)
            return
        }
        let topo = rtTopologyHash()
        // Recovery gate: the scene is back under the size caps, so the only
        // reason RT could still be off is the thrash trip. Re-arm it once the
        // topology has genuinely settled — a scene that is STILL churning (a
        // drag in progress, Forest's per-frame leaf geometry) never gets here
        // because its hash moves every frame and the counter keeps resetting.
        if rtAutoDisabled {
            let previousTrip = rts.rtGuard.trip?.rawValue ?? "?"
            guard rts.rtGuard.admitsRetry(topologyHash: topo) else {
                rts.rtTLASActive = false
                return
            }
            Self.log.info("""
                Illuminatorama RT re-armed — topology held for \
                \(IlluminatoramaRTHangGuard.recoveryStableFrames) frames after a \
                \(previousTrip) trip.
                """)
            rts.rtConsecutiveRebuilds = 0
            // Fall through: `topo` will differ from `rtTLASTopologyHash` (or the
            // TLAS is nil), so the rebuild branch below builds it once.
        }
        if topo != rts.rtTLASTopologyHash || rts.rtTLAS == nil {
            // Hang-proof gate 2 (rebuild-thrash): topology changing every frame
            // (GPU-fed / regenerated geometry — e.g. Forest's LeafField + bark +
            // swaying branches) ⇒ a full rebuild every frame ⇒ hang + faulted-AS
            // magenta. Allow a few legitimate settle frames, then disable.
            rts.rtConsecutiveRebuilds += 1
            if rts.rtConsecutiveRebuilds > Self.rtRebuildThrashLimit {
                tripRTGuard(.topologyThrashing, """
                    TLAS rebuilt \(rts.rtConsecutiveRebuilds) frames running (topology \
                    thrashing: GPU-fed / animated-topology / actively-edited scene). \
                    Retrying after \(IlluminatoramaRTHangGuard.recoveryStableFrames) settled frames
                    """)
                return
            }
            rebuildRTAccel()
            rts.rtTLASTopologyHash = topo
        } else {
            rts.rtConsecutiveRebuilds = 0   // stable topology this frame → reset
            // DH-0622 — cards otherwise bake only in `rebuildRTAccel`, i.e. on a topology change,
            // and turning the cache on is not one: a host that enables it over a live TLAS (a still
            // switching it on at capture) would refit forever and never get a card. Bake once per
            // (topology, budget); a scene over budget records its skip and does not retry.
            let surfBakeKey = (topo &* 31 &+ surfaceCacheMaxTriangles) &* 31 &+ surfaceCachePerTriangleTileSize
            if surfaceCacheEnabled, scs.surfCacheBakeKey != surfBakeKey {
                buildGroupedSurfaceSoup(total: instances.count)
                scs.surfCacheBakeKey = surfBakeKey
            }
            // PERF (static-scene skip): the TLAS is world-space — a camera-only
            // frame refit it into a byte-identical AS while rewriting every
            // instance descriptor on the CPU first. When the opaque instances
            // AND glass held since last frame, the TLAS already encodes exactly
            // this content, so skip both the descriptor rewrite and the refit
            // encode. Conservative outs: any curve sets (wind-swayed control
            // points) or deforming BLASes (GPU-fed vertices the CPU compare
            // can't see) force the refit every frame, as before.
            if gbs.instanceStableFrames >= 1, gls.glassStableFrames >= 1,
               rts.rtCurveSets.isEmpty, !rts.rtHasDeformingBLAS, rts.rtTLASActive {
                staticSkipStats.tlasRefitsSkipped += 1
                return
            }
            // Phase B — re-fit deforming BLASes from this frame's live vertices
            // BEFORE the TLAS refit so the RT passes trace current geometry.
            refitDeformingBLAS(cb)
            refitRTAccel(cb)
        }
    }

    /// Curve primitives (#60 item 7) — adopt the process-wide registry when its
    /// version moved: copy the sets, rebuild the pooled GPU buffers, and (on
    /// first use) compile the `kRTCurvesEnabled` kernel variants. The topology
    /// hash includes the sync version, so the caller's rebuild branch fires
    /// next. Curve support requires BOTH variant pipelines — if either fails
    /// to compile, the sets are dropped (a TLAS containing curves traced by a
    /// triangle-contract intersector is undefined behaviour, never shippable).
    func syncCurveRegistry() {
        let reg = IlluminatoramaCurveRegistry.shared
        guard reg.version != rts.rtCurveSyncVersion else { return }
        rts.rtCurveSyncVersion = reg.version
        rts.rtCurveSets = device.supportsRaytracing ? reg.sets : []
        if !rts.rtCurveSets.isEmpty {
            let cache = engine.pipelineCache
            var on = true
            if rts.rtTLASCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                rts.rtTLASCurvePipeline = cache.pipelineState(
                    name: "illumi_rt_lighting_tlas", device: device,
                    constants: cv, variantKey: "curves")
            }
            if scs.surfCacheTLASCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                scs.surfCacheTLASCurvePipeline = cache.pipelineState(
                    name: "illumi_surfcache_update_tlas", device: device,
                    constants: cv, variantKey: "curves")
            }
            if rts.rtTLASCurvePipeline == nil || scs.surfCacheTLASCurvePipeline == nil {
                Self.log.error("Curve RT: kRTCurvesEnabled pipeline variant failed; dropping \(self.rts.rtCurveSets.count) curve set(s).")
                rts.rtCurveSets = []
            }
        }
        buildCurvePools()
    }

    /// Function-constant index for `kRTCurvesEnabled` in
    /// IlluminatoramaRTInstanced.metal + IlluminatoramaSurfaceCache.metal.
    static let curveFunctionConstantIndex = 30

    /// Pool every adopted set's control points (packed float3 — stride 12, the
    /// layout both the curve BLAS descriptor and the shading kernels read),
    /// radii, and segment indices (made ABSOLUTE into the pool) into shared
    /// buffers, plus the per-set `RTCurveSetData` records. Bake-time only.
    func buildCurvePools() {
        rts.rtCurvePoolPoints = nil; rts.rtCurvePoolRadii = nil; rts.rtCurvePoolSegments = nil
        rts.rtCurveSetDataBuffer = nil; rts.rtCurveSetSegmentBase = []
        rts.rtCurvePoolPointCount = 0
        guard !rts.rtCurveSets.isEmpty else { return }
        var pts: [Float] = []          // 3 floats per control point (packed)
        var radii: [Float] = []
        var segs: [UInt32] = []
        var data: [RTCurveSetData] = []
        for set in rts.rtCurveSets {
            let pointBase = UInt32(radii.count)
            rts.rtCurveSetSegmentBase.append(segs.count)
            data.append(RTCurveSetData(
                m0: set.transform.columns.0, m1: set.transform.columns.1,
                m2: set.transform.columns.2, m3: set.transform.columns.3,
                albedoRoughness: SIMD4(set.material.albedo, set.material.roughness),
                emissionPad: SIMD4(set.material.emission, 0),
                meta: SIMD4<UInt32>(UInt32(segs.count), 0, 0, 0)))
            for p in set.controlPoints { pts.append(p.x); pts.append(p.y); pts.append(p.z) }
            radii.append(contentsOf: set.radii)
            segs.append(contentsOf: set.segmentIndices.map { $0 + pointBase })
        }
        guard let pb = device.makeBuffer(bytes: pts,
                  length: MemoryLayout<Float>.stride * pts.count, options: .storageModeShared),
              let rb = device.makeBuffer(bytes: radii,
                  length: MemoryLayout<Float>.stride * radii.count, options: .storageModeShared),
              let sb = device.makeBuffer(bytes: segs,
                  length: MemoryLayout<UInt32>.stride * segs.count, options: .storageModeShared),
              let db = device.makeBuffer(bytes: data,
                  length: MemoryLayout<RTCurveSetData>.stride * data.count, options: .storageModeShared)
        else {
            Self.log.error("Curve RT: pool allocation failed; dropping curve sets.")
            rts.rtCurveSets = []; rts.rtCurveSetSegmentBase = []
            return
        }
        pb.label = "Illuminatorama.rt.curve.points"
        rb.label = "Illuminatorama.rt.curve.radii"
        sb.label = "Illuminatorama.rt.curve.segments"
        db.label = "Illuminatorama.rt.curve.setData"
        rts.rtCurvePoolPoints = pb; rts.rtCurvePoolRadii = rb
        rts.rtCurvePoolSegments = sb; rts.rtCurveSetDataBuffer = db
        rts.rtCurvePoolPointCount = radii.count
    }

    /// Phase B — re-fit every deforming mesh's BLAS in place from its live
    /// (repacked) vertex buffer, on the FRAME command buffer. Runs only in the
    /// refit branch (stable topology); the repack at the top of the frame already
    /// wrote this frame's positions into `mesh.vertexBuffer` (the same buffer the
    /// BLAS references), and Metal sequences the compute-write → AS-refit-read on
    /// the shared command buffer. No `waitUntilCompleted` — refit is cheap + async.
    func refitDeformingBLAS(_ cb: MTLCommandBuffer) {
        guard rts.rtHasDeformingBLAS else { return }
        // Dev A/B override: VIZ_ILLUMI_NO_BLAS_REFIT=1 freezes deforming BLASes at
        // their build-time pose (the pre-Phase-B behaviour) so the RT-traces-stale
        // -vs-current difference is directly checkable. Default: refit (on).
        if Self.noBLASRefit { return }
        guard let enc = cb.makeAccelerationStructureCommandEncoder() else { return }
        enc.label = "Illuminatorama.rt.blas.refit"
        for entry in rts.rtMeshCache.values {
            guard let refit = entry.refit else { continue }
            enc.refit(sourceAccelerationStructure: entry.blas, descriptor: refit.desc,
                      destinationAccelerationStructure: entry.blas,
                      scratchBuffer: refit.scratch, scratchBufferOffset: 0)
        }
        enc.endEncoding()
    }

    func packed4x3(_ m: simd_float4x4) -> MTLPackedFloat4x3 {
        func col(_ c: SIMD4<Float>) -> MTLPackedFloat3 {
            var p = MTLPackedFloat3(); p.x = c.x; p.y = c.y; p.z = c.z; return p
        }
        var p = MTLPackedFloat4x3()
        p.columns.0 = col(m.columns.0); p.columns.1 = col(m.columns.1)
        p.columns.2 = col(m.columns.2); p.columns.3 = col(m.columns.3)
        return p
    }

    /// The TLAS transform of an instance whose descriptor `mask` is `mask`.
    ///
    /// **A mask-0 instance (raster-only, or a group with no BLAS) is COLLAPSED to a point.** No
    /// ray ever intersects it, but its bounds still go into the instance BVH, and a ray pays for
    /// every node it enters before the mask is tested at the leaf. Rebuild already left an
    /// unslotted instance zeroed; the REFIT then wrote its real model matrix back over the zero
    /// (paired with BLAS 0 — whatever mesh that is), so after one refit the BVH built around a
    /// point held a box as big as that matrix. Daydream DH-1474: a yard planting's batched
    /// buckets carry one raster-only placeholder per bucket whose matrix bounds the WHOLE
    /// planting (the CPU group cull reads it); ~120 of them, refit to yard-sized boxes over the
    /// house, cost the settled frame's RT glass pass +21 ms (21 → 42 ms) and RT lighting +2 ms.
    /// The point sits 10 km below the scene rather than at the origin (usually inside the house),
    /// so the collapsed instances share a node with nothing a ray is heading for.
    func rtDescriptorTransform(_ m: simd_float4x4, mask: UInt32) -> MTLPackedFloat4x3 {
        guard mask == 0 else { return packed4x3(m) }
        var p = MTLPackedFloat4x3()
        p.columns.3.y = -1.0e4
        return p
    }
}
