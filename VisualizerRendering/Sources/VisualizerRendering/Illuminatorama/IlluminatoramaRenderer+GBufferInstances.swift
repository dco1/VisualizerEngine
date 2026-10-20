import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    // ── Upload helpers ────────────────────────────────────────────────────────

    func ensureBufferCapacity() {
        if instances.count > gbs.instanceCapacity {
            let newCap = max(instances.count, gbs.instanceCapacity * 2)
            let stride = MemoryLayout<IlluminatoramaInstance>.stride * newCap
            // Grow BOTH ping-pong buffers in lockstep — the previous-frame
            // buffer is now uninitialized for motion-vector purposes, so we
            // also force TAA back into first-frame mode.
            if let a = device.makeBuffer(length: stride, options: .storageModeShared),
               let b = device.makeBuffer(length: stride, options: .storageModeShared) {
                a.label = "Illuminatorama.instancesA"
                b.label = "Illuminatorama.instancesB"
                gbs.instanceBufferA = a
                gbs.instanceBufferB = b
                gbs.instanceCapacity = newCap
                pfx.taaNeedsFirstFrame = true
            }
        }
        // Grow-on-demand for the ring-buffered per-frame inputs. Reallocate the
        // WHOLE ring (all maxFramesInFlight slots) so every in-flight frame keeps
        // a same-size buffer — growing only the current slot would let an
        // in-flight frame bind the old (too-small) slot next time round the ring.
        // Slots are uploaded fresh each frame, so there is nothing to copy over.
        func growRing<T>(_ ring: inout [MTLBuffer], _ capacity: inout Int,
                         needed: Int, type: T.Type, label: String) {
            guard needed > capacity else { return }
            let newCap = max(needed, capacity * 2)
            var nr: [MTLBuffer] = []
            nr.reserveCapacity(ring.count)
            for i in 0..<ring.count {
                guard let nb = device.makeBuffer(
                    length: MemoryLayout<T>.stride * newCap, options: .storageModeShared
                ) else { return }   // alloc failed — keep the old ring, retry next frame
                nb.label = "\(label).\(i)"
                nr.append(nb)
            }
            ring = nr
            capacity = newCap
        }
        // Keep the parallel superquadric param ring at least as large as the
        // instance buffer (indexed by the same grouped instance id).
        growRing(&gbs.superquadricParamRing, &gbs.superquadricParamCapacity,
                 needed: instances.count, type: IlluminatoramaInstance.SuperquadricParam.self,
                 label: "Illuminatorama.superquadricParams")
        growRing(&shd.pointLightRing, &shd.pointLightCapacity,
                 needed: pointLights.count, type: IlluminatoramaPointLight.self,
                 label: "Illuminatorama.pointLights")
        growRing(&shd.spotLightRing, &shd.spotLightCapacity,
                 needed: spotLights.count, type: IlluminatoramaSpotLight.self,
                 label: "Illuminatorama.spotLights")
        growRing(&lts.areaLightRing, &lts.areaLightCapacity,
                 needed: areaLights.count, type: IlluminatoramaAreaLight.self,
                 label: "Illuminatorama.areaLights")
        growRing(&lts.extraDirectionalRing, &lts.extraDirectionalCapacity,
                 needed: extraDirectionals.count, type: IlluminatoramaDirectionalLight.self,
                 label: "Illuminatorama.extraDirectionals")
        growRing(&lts.interiorApertureRing, &lts.interiorApertureCapacity,
                 needed: interiorApertures.count, type: IlluminatoramaInteriorAperture.self,
                 label: "Illuminatorama.interiorApertures")
    }

    func uploadInstances() {
        // PERF (static-scene skip): at ≥2 stable frames BOTH ping-pong buffers
        // already hold exactly this content (each change is written twice, once
        // per buffer), `meshGroups` is the grouping of that same content, and
        // prev==current means zero motion vectors — which is precisely correct
        // for content that didn't move. Skipping avoids the per-frame regroup
        // dictionary, the full buffer rewrite, and the superquadric inverse
        // recomputes. Capacity growth can't be missed: growth implies the count
        // changed, which resets `instanceStableFrames`.
        if gbs.instanceStableFrames >= 2 { staticSkipStats.uploadsSkipped += 1; return }
        // After the ping-pong toggle at the top of render(), the buffer that
        // WAS previous last frame is now "current" — we overwrite it with
        // this frame's data. The other buffer (now "previous") still holds
        // last frame's data, which the G-buffer vertex shader reads for
        // motion-vector reconstruction.
        //
        // Phase 4.12 — write in GROUPED order so all instances of the same
        // `MeshKind` land in a contiguous run. That lets every pass that
        // iterates the scene's geometry (G-buffer, cascaded + spot
        // shadows) replace its per-instance loop with one
        // `drawIndexedPrimitives(..., instanceCount: group.count)` per
        // mesh kind. FloatingFlowers+ goes from ~1500 draw calls to ~10
        // without any shader changes — the vertex shader's
        // `instances[iid]` lookup walks the contiguous slice naturally
        // because we set the vertex/fragment buffer offsets per group.
        //
        // Grouping stability across frames: the host (`extractor`) walks
        // the scene graph in DFS order, so the same scene topology
        // produces the same first-seen ordering of mesh kinds + the same
        // intra-group instance order. That's what motion-vector TAA
        // relies on — `prevInstances[iid]` reads the same logical
        // instance as `instances[iid]` because both buffers used the
        // same grouping last → this frame.
        gbs.meshGroups.removeAll(keepingCapacity: true)
        gbs.meshGroupRange.removeAll(keepingCapacity: true)
        gbs.meshGroupsVersion &+= 1   // DH-0534: the cull bounds describe the old grouping now
        guard !instances.isEmpty else { return }

        // VEGETATION WIND, LIVENESS. `treeWindStrength` is a global, but as of 2026-08-17 the
        // displacement also needs a per-DRAW `windScale` (see the property's own note: without
        // it the gust reached any mesh whose tangents pointed along the sway axis, because a
        // tangent is a TBN basis and not a sway weight). A host that raises the strength and
        // never marks its foliage instances therefore gets NO wind and no error — the inert
        // feature this renderer keeps re-learning. Say so once, on the frame it happens.
        if treeWindStrength > 0, !pfx.warnedWindWithoutOptIn,
           !instances.contains(where: { $0.data.windScale > 0 }) {
            pfx.warnedWindWithoutOptIn = true
            print("[illuminatorama] treeWindStrength=\(treeWindStrength) but no instance sets "
                  + "windScale>0 — the vegetation wind is INERT. Set `windScale = 1` on the "
                  + "foliage draws whose vertices pack (swayWeight, phase, flutter) in tangent.")
        }

        // First pass — bucket by mesh kind, preserving first-seen order
        // so a stable scene gives a stable buffer layout.
        var byKind: [MeshKind: [Int]] = [:]
        byKind.reserveCapacity(8)
        var kindOrder: [MeshKind] = []
        kindOrder.reserveCapacity(8)
        for (srcIdx, ref) in instances.enumerated() {
            if byKind[ref.meshKind] == nil {
                byKind[ref.meshKind] = []
                kindOrder.append(ref.meshKind)
            }
            byKind[ref.meshKind, default: []].append(srcIdx)
        }

        // Second pass — write into the buffer in grouped order, building
        // the per-group draw recipe alongside.
        let dst = currentInstanceBuffer.contents().bindMemory(
            to: IlluminatoramaInstance.self, capacity: instances.count
        )
        // Parallel superquadric param buffer, written in the SAME grouped order so
        // `params[iid]` lines up with `instances[iid]` in the impostor draw. Most
        // entries are unused (only impostor instances carry a shape); cheap to fill.
        let sqDst = superquadricParamBuffer.contents().bindMemory(
            to: IlluminatoramaInstance.SuperquadricParam.self, capacity: instances.count
        )
        var dstIdx = 0
        for kind in kindOrder {
            guard let srcIndices = byKind[kind] else { continue }
            let groupStart = dstIdx
            for srcIdx in srcIndices {
                let ref = instances[srcIdx]
                dst[dstIdx] = ref.data
                if let s = ref.superquadricShape {
                    let isEllipsoid = (abs(s.x - 1) < 1e-3 && abs(s.y - 1) < 1e-3)
                    sqDst[dstIdx] = IlluminatoramaInstance.SuperquadricParam(
                        invModel: simd_inverse(ref.data.modelMatrix),
                        shape: SIMD4(s.x, s.y, isEllipsoid ? 1 : 0, 0))
                } else {
                    sqDst[dstIdx] = IlluminatoramaInstance.SuperquadricParam(
                        invModel: matrix_identity_float4x4, shape: .zero)
                }
                dstIdx += 1
            }
            gbs.meshGroups.append(MeshDrawGroup(
                kind: kind, start: groupStart, count: srcIndices.count))
            gbs.meshGroupRange[kind] = groupStart ..< (groupStart + srcIndices.count)
        }
    }

    /// DH-0534 — rebuild `meshGroupCullBounds` when its inputs changed. Everything the CPU cannot
    /// see is UNBOUNDED and always draws: a host `onEncodeGPUInstances` kernel may overwrite ANY
    /// instance slot (so with the hook set, nothing is culled at all); `gpuRepackTasks` rewrite a
    /// kind's vertices (which also covers the deforming meshes behind `prevPosByKind`); and a mesh
    /// built on caller-owned buffers has no CPU-visible vertices (`cpuAuthoredVertices == false`).
    func computeCullBounds() {
        let hostWritesInstances = onEncodeGPUInstances != nil
        let key = CullBoundsKey(groupsVersion: gbs.meshGroupsVersion, wind: treeWindStrength,
                                hostWritesInstances: hostWritesInstances,
                                repackCount: fxs.gpuRepackTasks.count)
        if key == gbs.cullBoundsKey, gbs.meshGroupCullBounds.count == gbs.meshGroups.count { return }
        gbs.cullBoundsKey = key
        gbs.meshGroupCullBounds.removeAll(keepingCapacity: true)
        let total = gbs.meshGroups.last.map { $0.start + $0.count } ?? 0
        guard total > 0 else { return }
        let repacked = Set(fxs.gpuRepackTasks.map(\.kind))
        let inst = currentInstanceBuffer.contents()
            .bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        for group in gbs.meshGroups {
            guard !hostWritesInstances, !repacked.contains(group.kind),
                  let mesh = meshes[group.kind], mesh.cpuAuthoredVertices,
                  let local = mesh.boundingSphere else {
                gbs.meshGroupCullBounds.append(.unbounded)
                continue
            }
            var bounds = IlluminatoramaCullBounds.empty
            var bounded = true
            for i in group.start ..< group.start + group.count {
                guard let sphere = IlluminatoramaCullBounds.worldSphere(
                    local: local, instance: inst[i], treeWindStrength: treeWindStrength,
                    windMax: { mesh.windAttributeMax }) else { bounded = false; break }
                bounds.formUnion(center: sphere.center, radius: sphere.radius)
            }
            gbs.meshGroupCullBounds.append(bounded ? bounds : .unbounded)
        }
    }

    /// The clip volume a pass culls against, or nil when culling is off (nil ⇒ draw everything).
    func cullVolume(_ vp: simd_float4x4) -> IlluminatoramaClipVolume? {
        visibilityCullingEnabled ? IlluminatoramaClipVolume(vp) : nil
    }

    /// DH-0534 — true when group `gi` cannot produce a fragment in a pass rasterising with `clip`.
    /// Counts the group into `drawn` or `culled` either way, so the stats are complete.
    @inline(__always)
    func shouldCull(_ gi: Int, _ group: MeshDrawGroup, _ mesh: IlluminatoramaMesh,
                            _ clip: IlluminatoramaClipVolume?,
                            drawn: WritableKeyPath<IlluminatoramaVisibilityCullStats, Int>,
                            culled: WritableKeyPath<IlluminatoramaVisibilityCullStats, Int>) -> Bool {
        if let clip, gi < gbs.meshGroupCullBounds.count, clip.excludes(gbs.meshGroupCullBounds[gi]) {
            visibilityCullStats[keyPath: culled] += 1
            visibilityCullStats.trianglesCulled += (mesh.indexCount / 3) * group.count
            return true
        }
        visibilityCullStats[keyPath: drawn] += 1
        return false
    }
}
