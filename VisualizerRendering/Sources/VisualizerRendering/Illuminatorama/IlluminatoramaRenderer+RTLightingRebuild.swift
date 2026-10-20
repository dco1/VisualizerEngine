import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Full rebuild: build a BLAS + object-normal slot for every not-yet-cached
    /// mesh (one setup-time command buffer), assemble the concatenated normal
    /// buffer + per-instance descriptors/data in grouped order, and build the
    /// TLAS. Runs on a topology change.
    func rebuildRTAccel() {
        rts.rtTLASActive = false
        // The scene's mesh set just changed — release every cached entry whose mesh is gone
        // BEFORE looking anything up below, so the cache never outlives the addresses it is
        // keyed by (see `RTMeshEntry`). Also refreshes `rtHasDeformingBLAS`.
        pruneRTMeshCache()
        // AAA glass (#60): the canonical flattened glass list in TLAS-append order.
        // Glass instances ride the SAME TLAS as the opaque/curve instances (so glass
        // refracts/reflects real geometry and other glass) but carry mask 0x02, so
        // the deferred RT lighting / surface-cache passes (which trace with mask
        // 0x01) never see them. They're appended AFTER opaque + curve instances;
        // `rtGlassInstanceBase` is the first glass instance_id.
        let glassFlat: [(kind: MeshKind, inst: IlluminatoramaGlassInstance)] =
            (rtGlassEnabled ? flattenGlass() : []).flatMap { g in g.insts.map { (g.kind, $0) } }
        // (a) Build BLAS + cache object normals for any newly-seen mesh.
        // Deforming (GPU-fed / DynamicMesh) kinds get a refit-capable BLAS so the
        // per-frame `refitDeformingBLAS` can track their moving vertices (Phase B).
        let deformingKinds = Set(fxs.gpuRepackTasks.map(\.kind))
        var pending: [(MTLAccelerationStructure, MTLPrimitiveAccelerationStructureDescriptor, MTLBuffer)] = []
        for group in gbs.meshGroups {
            guard let mesh = meshes[group.kind], mesh.indexCount >= 3 else { continue }
            let mid = ObjectIdentifier(mesh)
            if let cached = rts.rtMeshCache[mid] {
                // Ownership makes this impossible; assert it anyway, because when it was
                // possible it silently traced the wrong geometry through every glass pane.
                if cached.mesh !== mesh { rtStaleBLASReuseCount += 1 }
                continue
            }
            let deforming = deformingKinds.contains(group.kind)
            let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
            geom.vertexBuffer = mesh.vertexBuffer
            geom.vertexStride = MemoryLayout<IlluminatoramaVertex>.stride
            geom.vertexFormat = .float3
            geom.indexBuffer = mesh.indexBuffer
            geom.indexType = mesh.indexType
            geom.triangleCount = mesh.indexCount / 3
            // #60 item 6 — bake this mesh's per-triangle card-frame UVs into the
            // BLAS `primitive_data` (one `IlluminatoramaPrimUV` per primitive,
            // mesh-invariant). A TLAS hit then reads its surface-cache UVs from the
            // AS payload (`res.primitive_data`) instead of the per-instance global
            // `triUVa`/`triUVc` side buffers. Per-triangle path only (charts use a
            // non-invariant parameterisation) and only when the cache is on. Metal
            // COPIES this buffer into the AS at build time, so it only needs to live
            // through the build below — `pending`→`d`→`geom` retains it until then.
            //
            // DEFORMING kinds are excluded: their soup comes from the UN-sanitised
            // `liveDeformingObjectSoup`, so a degenerate/out-of-range triangle would
            // drop from the global soup (shifting its post-drop frame-A/B parity)
            // while the raw-count BLAS table would not — a silent wrong-tile-half.
            // Their BLAS keeps no primitive data → the shader null-check falls back
            // to the (correct, post-drop) side-buffer UVs. They also re-light every
            // frame (α=1), so the cache-read saving matters least for them anyway.
            if bakePrimUVFold && !deforming {
                let uvs = Self.perMeshCardUVs(triangleCount: mesh.indexCount / 3)
                let stride = MemoryLayout<IlluminatoramaPrimUV>.stride   // 24 B (3×float2)
                if !uvs.isEmpty,
                   let pdb = device.makeBuffer(bytes: uvs, length: stride * uvs.count,
                                               options: .storageModeShared) {
                    pdb.label = "Illuminatorama.rt.blas.primUV"
                    geom.primitiveDataBuffer = pdb
                    geom.primitiveDataBufferOffset = 0
                    geom.primitiveDataStride = stride
                    geom.primitiveDataElementSize = stride
                }
            }
            let d = MTLPrimitiveAccelerationStructureDescriptor()
            d.geometryDescriptors = [geom]
            // `.refit` lets the BLAS be re-fit in place each frame from the live
            // vertex buffer (fixed topology). Only deforming meshes need it; static
            // meshes keep the default build (smaller, no refit scratch retained).
            if deforming { d.usage = .refit }
            let sizes = device.accelerationStructureSizes(descriptor: d)
            guard let blas = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                                   options: .storageModePrivate) else { continue }
            blas.label = deforming ? "Illuminatorama.rt.blas.deforming" : "Illuminatorama.rt.blas"
            var entry = RTMeshEntry(mesh: mesh, blas: blas,
                                    normals: mesh.objectFaceNormals(),
                                    uvs: mesh.objectFaceUVs(), refit: nil)
            pending.append((blas, d, scratch))
            // Retain the descriptor + a refit-sized scratch so the per-frame refit
            // can re-encode without re-deriving sizes. Refit scratch can differ from
            // build scratch; size it explicitly.
            if deforming, let rscratch = device.makeBuffer(
                    length: max(sizes.refitScratchBufferSize, 16),
                    options: .storageModePrivate) {
                entry.refit = (desc: d, scratch: rscratch)
            }
            rts.rtMeshCache[mid] = entry
        }
        // (a2) Curve BLASes (#60 item 7) — one per registered set, rebuilt on
        // every topology change (cheap: that's the whole point of curve
        // primitives vs triangle tubes). Each BLAS reads its segment slice of
        // the pooled buffers via `indexBufferOffset`; segment index values are
        // absolute into the pool, so `controlPointBuffer` is the whole pool.
        rts.rtCurveBLASList.removeAll(keepingCapacity: true)
        if !rts.rtCurveSets.isEmpty,
           let cpts = rts.rtCurvePoolPoints, let crad = rts.rtCurvePoolRadii,
           let cseg = rts.rtCurvePoolSegments {
            for (s, set) in rts.rtCurveSets.enumerated() where s < rts.rtCurveSetSegmentBase.count {
                let geom = MTLAccelerationStructureCurveGeometryDescriptor()
                geom.controlPointBuffer = cpts
                geom.controlPointCount = rts.rtCurvePoolPointCount
                geom.controlPointStride = 12       // packed float3
                geom.controlPointFormat = .float3
                geom.radiusBuffer = crad
                geom.radiusFormat = .float
                geom.radiusStride = MemoryLayout<Float>.stride
                geom.indexBuffer = cseg
                geom.indexBufferOffset = rts.rtCurveSetSegmentBase[s] * MemoryLayout<UInt32>.stride
                geom.indexType = .uint32
                geom.segmentCount = set.segmentIndices.count
                geom.segmentControlPointCount = 4
                geom.curveType = .round            // exact 3D normals (branches)
                geom.curveBasis = .catmullRom
                geom.curveEndCaps = .sphere
                geom.opaque = true
                let d = MTLPrimitiveAccelerationStructureDescriptor()
                d.geometryDescriptors = [geom]
                let sizes = device.accelerationStructureSizes(descriptor: d)
                guard let blas = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                      let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                                       options: .storageModePrivate) else { continue }
                blas.label = "Illuminatorama.rt.blas.curve.\(set.label)"
                rts.rtCurveBLASList.append(blas)
                pending.append((blas, d, scratch))
            }
        }
        // (a3) Glass BLASes (#60 AAA glass) — one per distinct glass mesh kind not
        // already cached (a glass kind may be shared with an opaque mesh, e.g.
        // `.box`; `rtMeshCache` dedupes). Static (no refit / no primUV bake): glass
        // doesn't participate in the surface cache as a hit surface.
        for (kind, _) in glassFlat {
            guard let mesh = meshes[kind], mesh.indexCount >= 3 else { continue }
            let mid = ObjectIdentifier(mesh)
            if let cached = rts.rtMeshCache[mid] {
                // Ownership makes this impossible; assert it anyway, because when it was
                // possible it silently traced the wrong geometry through every glass pane.
                if cached.mesh !== mesh { rtStaleBLASReuseCount += 1 }
                // A mesh first cached as OPAQUE (a shared kind) never read its corner normals.
                if cached.cornerNormals.isEmpty {
                    rts.rtMeshCache[mid]?.cornerNormals = mesh.objectCornerNormals()
                }
                continue
            }
            let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
            geom.vertexBuffer = mesh.vertexBuffer
            geom.vertexStride = MemoryLayout<IlluminatoramaVertex>.stride
            geom.vertexFormat = .float3
            geom.indexBuffer = mesh.indexBuffer
            geom.indexType = mesh.indexType
            geom.triangleCount = mesh.indexCount / 3
            let d = MTLPrimitiveAccelerationStructureDescriptor()
            d.geometryDescriptors = [geom]
            let sizes = device.accelerationStructureSizes(descriptor: d)
            guard let blas = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                                   options: .storageModePrivate) else { continue }
            blas.label = "Illuminatorama.rt.blas.glass"
            rts.rtMeshCache[mid] = RTMeshEntry(mesh: mesh, blas: blas,
                                           normals: mesh.objectFaceNormals(),
                                           uvs: mesh.objectFaceUVs(), refit: nil,
                                           cornerNormals: mesh.objectCornerNormals())
            pending.append((blas, d, scratch))
        }
        if !pending.isEmpty {
            guard let cmd = commandQueue.makeCommandBuffer(),
                  let enc = cmd.makeAccelerationStructureCommandEncoder() else { return }
            for (blas, d, scratch) in pending {
                enc.build(accelerationStructure: blas, descriptor: d,
                          scratchBuffer: scratch, scratchBufferOffset: 0)
            }
            enc.endEncoding(); cmd.commit()
            cmd.waitUntilCompleted()   // gpu-ok: one-time BLAS build (per newly-seen mesh / per-topology curve set)
            Self.recordASBuildError(cmd, "rebuildRTAccel.blas")
        }

        // (b) Assemble BLAS list + concat normals + per-mesh slot for this topology.
        rts.rtBLASList.removeAll(keepingCapacity: true)
        rts.rtResidentBuffers.removeAll(keepingCapacity: true)
        var concat: [SIMD4<Float>] = []
        // Per-triangle mesh UVs, concatenated in LOCKSTEP with `concat` (3 entries
        // per normal). A mesh whose UVs couldn't be read (GPU-private vertices) is
        // padded with zeros so `normalBase * 3` stays a valid index for every later
        // mesh — a short array here would shift every subsequent mesh's UVs.
        var concatUV: [SIMD2<Float>] = []
        func appendMesh(_ cached: RTMeshEntry) {
            concat.append(contentsOf: cached.normals)
            let want = cached.normals.count * 3
            concatUV.append(contentsOf: cached.uvs.prefix(want))
            if cached.uvs.count < want {
                concatUV.append(contentsOf: repeatElement(.zero, count: want - cached.uvs.count))
            }
        }
        var slot: [ObjectIdentifier: (asIndex: Int, normalBase: Int)] = [:]
        for group in gbs.meshGroups {
            guard let mesh = meshes[group.kind] else { continue }
            let mid = ObjectIdentifier(mesh)
            if slot[mid] != nil { continue }
            guard let cached = rts.rtMeshCache[mid] else { continue }
            slot[mid] = (rts.rtBLASList.count, concat.count)
            rts.rtBLASList.append(cached.blas)
            rts.rtResidentBuffers.append(mesh.vertexBuffer); rts.rtResidentBuffers.append(mesh.indexBuffer)
            appendMesh(cached)
        }
        // Glass meshes join the same BLAS list + concat normals (a glass hit's
        // normal resolves through `objNormal[normalBase + prim]`, same as opaque).
        for (kind, _) in glassFlat {
            guard let mesh = meshes[kind] else { continue }
            let mid = ObjectIdentifier(mesh)
            if slot[mid] != nil { continue }
            guard let cached = rts.rtMeshCache[mid] else { continue }
            slot[mid] = (rts.rtBLASList.count, concat.count)
            rts.rtBLASList.append(cached.blas)
            rts.rtResidentBuffers.append(mesh.vertexBuffer); rts.rtResidentBuffers.append(mesh.indexBuffer)
            appendMesh(cached)
        }
        guard !rts.rtBLASList.isEmpty, !concat.isEmpty else { return }
        // Glass corner normals: one slice per distinct glass mesh, base stored +1 in each glass
        // instance's row (see `rtGlassCornerNormalBuffer`).
        var cornerConcat: [SIMD4<Float>] = []
        var cornerBase: [ObjectIdentifier: Int] = [:]
        for (kind, _) in glassFlat {
            guard let mesh = meshes[kind] else { continue }
            let mid = ObjectIdentifier(mesh)
            guard cornerBase[mid] == nil, let cached = rts.rtMeshCache[mid],
                  !cached.cornerNormals.isEmpty,
                  cached.cornerNormals.count == cached.normals.count * 3 else { continue }
            cornerBase[mid] = cornerConcat.count
            cornerConcat.append(contentsOf: cached.cornerNormals)
        }
        gls.rtGlassCornerNormalBuffer = cornerConcat.isEmpty ? nil : device.makeBuffer(
            bytes: cornerConcat, length: MemoryLayout<SIMD4<Float>>.stride * cornerConcat.count,
            options: .storageModeShared)
        gls.rtGlassCornerNormalBuffer?.label = "Illuminatorama.rt.glassCornerNormal"
        rts.rtObjNormalBuffer = device.makeBuffer(
            bytes: concat, length: MemoryLayout<SIMD4<Float>>.stride * concat.count,
            options: .storageModeShared)
        rts.rtObjNormalCount = concat.count
        rts.rtObjUVBuffer = device.makeBuffer(
            bytes: concatUV, length: MemoryLayout<SIMD2<Float>>.stride * max(1, concatUV.count),
            options: .storageModeShared)
        rts.rtObjUVBuffer?.label = "Illuminatorama.rt.objUV"
        rts.rtObjUVCount = concatUV.count
        // Curve pool buffers must stay resident for the intersector (the curve
        // BLASes reference them), and the shading kernels read them at hits.
        if !rts.rtCurveBLASList.isEmpty,
           let cpts = rts.rtCurvePoolPoints, let crad = rts.rtCurvePoolRadii,
           let cseg = rts.rtCurvePoolSegments {
            rts.rtResidentBuffers.append(cpts)
            rts.rtResidentBuffers.append(crad)
            rts.rtResidentBuffers.append(cseg)
        }

        // (c) Per-instance descriptors + data in GROUPED order (== currentInstanceBuffer).
        // Curve-set instances (#60 item 7) append after the triangle instances.
        let total = instances.count
        let curveCount = rts.rtCurveBLASList.count
        let glassCount = glassFlat.count
        let totalInstances = total + curveCount + glassCount
        ensureRTInstanceBuffers(capacity: totalInstances)
        guard let descBuf = rts.rtInstanceDescBuffer, let dataBuf = rts.rtInstanceDataBuffer else { return }
        // Zero first so any unslotted instance has mask=0 (never intersected). The
        // data buffer is zeroed across ALL instances (incl. glass) so a glass hit's
        // RTInstanceData (filled below) and the curve gap are well-defined.
        memset(descBuf.contents(), 0, MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * totalInstances)
        memset(dataBuf.contents(), 0, MemoryLayout<RTInstanceData>.stride * totalInstances)
        let instPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let descPtr = descBuf.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: totalInstances)
        let dataPtr = dataBuf.contents().bindMemory(to: RTInstanceData.self, capacity: totalInstances)
        for group in gbs.meshGroups {
            guard let mesh = meshes[group.kind], let s = slot[ObjectIdentifier(mesh)] else { continue }
            for k in 0..<group.count {
                let i = group.start + k
                guard i < total else { break }
                let inst = instPtr[i]
                var desc = MTLAccelerationStructureInstanceDescriptor()
                desc.accelerationStructureIndex = UInt32(s.asIndex)
                // Three states — see the `rtMask*` constants. `rtExclude` 1 is the
                // original raster-only case (#60 item 7): present in the TLAS
                // layout, never intersected, its RT representation supplied
                // elsewhere. `rtExclude` 2 is the INVISIBLE OCCLUDER (C3): real to
                // transport rays, invisible to camera-visible ones. Everything else
                // is an ordinary traced opaque instance (AAA glass uses 0x02), so
                // rays tracing 0x01 hit exactly what they hit before glass joined
                // the TLAS.
                switch inst.rtExclude {
                case IlluminatoramaInstance.rtExcludeFully:
                    desc.mask = 0x00
                case IlluminatoramaInstance.rtInvisibleOccluder:
                    desc.mask = Self.rtMaskInvisibleOccluder
                default:
                    desc.mask = Self.rtMaskTraced
                }
                desc.options = .opaque
                desc.intersectionFunctionTableOffset = 0
                desc.transformationMatrix = packed4x3(inst.modelMatrix)
                descPtr[i] = desc
                dataPtr[i] = Self.rtInstanceRow(inst, normalBase: Float(s.normalBase))
            }
        }
        // (c2) Curve-set instance descriptors (#60 item 7) — one per set,
        // static rigid placement captured at registration; a hit's
        // `instance_id - total` is the set index into `rtCurveSetDataBuffer`.
        for (c, set) in rts.rtCurveSets.enumerated() where c < curveCount {
            var desc = MTLAccelerationStructureInstanceDescriptor()
            desc.accelerationStructureIndex = UInt32(rts.rtBLASList.count + c)
            desc.mask = 0x01   // opaque/curve mask (glass = 0x02)
            desc.options = .opaque
            desc.intersectionFunctionTableOffset = 0
            desc.transformationMatrix = packed4x3(set.transform)
            descPtr[total + c] = desc
        }
        rts.rtCurveInstanceCount = curveCount

        // (c3) Glass instance descriptors + data + the per-glass material buffer
        // (#60 AAA glass). Appended after opaque + curve; `rtGlassInstanceBase` is
        // the first glass instance_id. Mask 0x02 so only glass-pass rays (which
        // trace 0x03 = opaque+glass) intersect them. The bounce loop reads the hit's
        // material from `rtGlassDataBuffer[instance_id - base]` and its normal from
        // the shared `objNormal` via this RTInstanceData (albedo carries the tint).
        gls.rtGlassInstanceBase = total + curveCount
        gls.rtGlassInstanceCount = glassCount
        if glassCount > 0 {
            var glassData = [IlluminatoramaRTGlassData](
                repeating: IlluminatoramaRTGlassData(tintIor: .zero, rdrf: .zero, dispersionPad: .zero),
                count: glassCount)
            for (gi, entry) in glassFlat.enumerated() {
                let i = gls.rtGlassInstanceBase + gi
                guard let mesh = meshes[entry.kind], let s = slot[ObjectIdentifier(mesh)] else { continue }
                let inst = entry.inst
                var desc = MTLAccelerationStructureInstanceDescriptor()
                desc.accelerationStructureIndex = UInt32(s.asIndex)
                desc.mask = 0x02
                desc.options = .opaque   // opaque AS hit (the dielectric BSDF is shaded in the glass fs)
                desc.intersectionFunctionTableOffset = 0
                desc.transformationMatrix = packed4x3(inst.modelMatrix)
                descPtr[i] = desc
                let cb = cornerBase[ObjectIdentifier(mesh)].map { Float($0 + 1) } ?? 0
                dataPtr[i] = Self.rtGlassInstanceRow(inst, normalBase: Float(s.normalBase), cornerBase: cb)
                glassData[gi] = IlluminatoramaRTGlassData(
                    tintIor: inst.tintIor, rdrf: inst.rdrf, dispersionPad: inst.dispersionPad)
            }
            let glen = MemoryLayout<IlluminatoramaRTGlassData>.stride * glassCount
            if (gls.rtGlassDataBuffer?.length ?? 0) < glen {
                gls.rtGlassDataBuffer = device.makeBuffer(length: glen, options: .storageModeShared)
                gls.rtGlassDataBuffer?.label = "Illuminatorama.rt.glassData"
            }
            glassData.withUnsafeBytes { src in
                _ = memcpy(gls.rtGlassDataBuffer!.contents(), src.baseAddress!, glen)
            }
        }

        // (d) Build the TLAS (allocate / grow as needed).
        let tdesc = MTLInstanceAccelerationStructureDescriptor()
        tdesc.instanceDescriptorBuffer = descBuf
        tdesc.instanceCount = totalInstances
        tdesc.instancedAccelerationStructures = rts.rtBLASList + rts.rtCurveBLASList
        tdesc.usage = .refit
        let sizes = device.accelerationStructureSizes(descriptor: tdesc)
        if rts.rtTLAS == nil || rts.rtTLASCapacity < totalInstances {
            rts.rtTLAS = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
            rts.rtTLASScratch = device.makeBuffer(
                length: max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize, 16),
                options: .storageModePrivate)
            rts.rtTLAS?.label = "Illuminatorama.rt.tlas"
            rts.rtTLASCapacity = totalInstances
        }
        guard let tlas = rts.rtTLAS, let scratch = rts.rtTLASScratch,
              let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeAccelerationStructureCommandEncoder() else { return }
        enc.build(accelerationStructure: tlas, descriptor: tdesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding(); cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: topology-change TLAS rebuild (not per-frame for stable scenes)
        Self.recordASBuildError(cmd, "rebuildRTAccel.tlas")
        rts.rtTLASInstanceCount = totalInstances
        rts.rtTLASActive = true

        // P1c — surface cache + TLAS coexistence. Bake a per-instance world-space
        // soup in the SAME grouped order this TLAS enumerates instances, so a
        // hit's `soupTriBase[instance_id] + primitive_id` indexes the soup-ordered
        // per-triangle cards. The soup also feeds the primitive `rtAccel` the
        // cache-UPDATE pass traces. Runs only on topology change (not per refit),
        // so an animated scene's moved geometry reads bounded-stale cache — lighting
        // still animates via the TLAS refit. Skipped (TLAS unaffected) past the cap.
        if surfaceCacheEnabled {
            buildGroupedSurfaceSoup(total: total)
            scs.surfCacheBakeKey = (rtTopologyHash() &* 31 &+ surfaceCacheMaxTriangles) &* 31 &+ surfaceCachePerTriangleTileSize
        }
    }

    /// In-place TLAS refit on the frame's command buffer — same topology, only
    /// transforms/materials changed. No vertex transform, no rebuild, no wait.
    func refitRTAccel(_ cb: MTLCommandBuffer) {
        // AAA glass (#60): glass instances animate too (e.g. orbiting lenses), so
        // the refit must include them — and the instance-count check has to count
        // glass, or a glass scene would rebuild every frame (thrash → auto-disable).
        let glassFlat: [(kind: MeshKind, inst: IlluminatoramaGlassInstance)] =
            (rtGlassEnabled ? flattenGlass() : []).flatMap { g in g.insts.map { (g.kind, $0) } }
        guard let tlas = rts.rtTLAS, let scratch = rts.rtTLASScratch,
              let descBuf = rts.rtInstanceDescBuffer, let dataBuf = rts.rtInstanceDataBuffer,
              rts.rtTLASInstanceCount == instances.count + rts.rtCurveInstanceCount + glassFlat.count,
              glassFlat.count == gls.rtGlassInstanceCount,
              !rts.rtBLASList.isEmpty else {
            rebuildRTAccel(); return   // topology drifted — full rebuild
        }
        // Curve instances keep their rebuild-time descriptors (static rigid
        // placement); the triangle (opaque) + glass instances' transforms refresh.
        let total = instances.count
        let cap = rts.rtTLASInstanceCount
        let instPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let descPtr = descBuf.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: cap)
        let dataPtr = dataBuf.contents().bindMemory(to: RTInstanceData.self, capacity: cap)
        // Update transforms + normal matrices in place; asIndex + normalBase
        // (set during rebuild) are unchanged for a stable topology.
        for group in gbs.meshGroups {
            guard meshes[group.kind] != nil else { continue }
            for k in 0..<group.count {
                let i = group.start + k
                guard i < total else { break }
                let inst = instPtr[i]
                descPtr[i].transformationMatrix = packed4x3(inst.modelMatrix)
                // Whole row through the ONE packer. Writing `nrm0`/`nrm1` straight
                // from the normal matrix — as this did — wiped the albedo-atlas slice
                // and the light-layer mask packed into their `.w` lanes, so from the
                // second frame after a rebuild every re-shaded secondary hit lost its
                // texture and its light masking. `normalBase` is topology, so it comes
                // back out of the row.
                dataPtr[i] = Self.rtInstanceRow(inst, normalBase: dataPtr[i].albedoTriBase.w)
            }
        }
        // Glass instances [rtGlassInstanceBase, …): refresh transform + normal +
        // per-glass material (tint/ior may animate). Same flatten order as rebuild.
        if gls.rtGlassInstanceCount > 0, let gdata = gls.rtGlassDataBuffer {
            let gPtr = gdata.contents().bindMemory(to: IlluminatoramaRTGlassData.self, capacity: gls.rtGlassInstanceCount)
            for (gi, entry) in glassFlat.enumerated() {
                let i = gls.rtGlassInstanceBase + gi
                guard i < cap else { break }
                let inst = entry.inst
                descPtr[i].transformationMatrix = packed4x3(inst.modelMatrix)
                dataPtr[i] = Self.rtGlassInstanceRow(inst, normalBase: dataPtr[i].albedoTriBase.w,
                                                     cornerBase: dataPtr[i].emissionPad.w)
                gPtr[gi] = IlluminatoramaRTGlassData(
                    tintIor: inst.tintIor, rdrf: inst.rdrf, dispersionPad: inst.dispersionPad)
            }
        }
        let tdesc = MTLInstanceAccelerationStructureDescriptor()
        tdesc.instanceDescriptorBuffer = descBuf
        tdesc.instanceCount = rts.rtTLASInstanceCount
        tdesc.instancedAccelerationStructures = rts.rtBLASList + rts.rtCurveBLASList
        tdesc.usage = .refit
        guard let enc = cb.makeAccelerationStructureCommandEncoder() else { return }
        enc.refit(sourceAccelerationStructure: tlas, descriptor: tdesc,
                  destinationAccelerationStructure: tlas, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        rts.rtTLASActive = true
    }

    func ensureRTInstanceBuffers(capacity: Int) {
        let descLen = MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * max(1, capacity)
        if (rts.rtInstanceDescBuffer?.length ?? 0) < descLen {
            rts.rtInstanceDescBuffer = device.makeBuffer(length: descLen, options: .storageModeShared)
            rts.rtInstanceDescBuffer?.label = "Illuminatorama.rt.instanceDesc"
        }
        let dataLen = MemoryLayout<RTInstanceData>.stride * max(1, capacity)
        if (rts.rtInstanceDataBuffer?.length ?? 0) < dataLen {
            rts.rtInstanceDataBuffer = device.makeBuffer(length: dataLen, options: .storageModeShared)
            rts.rtInstanceDataBuffer?.label = "Illuminatorama.rt.instanceData"
        }
    }
}
