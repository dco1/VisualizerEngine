import Metal
import simd

// GPU-DRIVEN SHADOW DRAW SUBMISSION (Daydream DH-1684) — the encode.
//
// A shadow pass that takes this path:
//   1. `encodeGPUShadowViews` (before the pass's render encoders) — one compute dispatch writes every
//      view's draw commands into the frame slot's ICB, culling each record against the view.
//   2. per view, inside its render encoder: the CPU loop draws only `table.fallback`, then
//      `executeGPUShadowView` runs the view's commands, one range per cull-mode bucket.
// Nothing about WHICH geometry lands in a map changes — see IlluminatoramaGPUDraws.metal.
// Lever: `gpuShadowDrawsEnabled` (`VIZ_ILLUMI_NO_GPU_DRAWS=1`). Gate:
// `HouseRenderBridgeGPUTests_GPUShadowDraws` (Daydream) — both paths, byte-compared shadow maps.
extension IlluminatoramaRenderer {

    /// Draw the sun cascades' and spot/area slices' shadow casters from GPU-written indirect
    /// command buffers (DH-1684). Off = the CPU loop encodes every draw, as before.
    public var gpuShadowDrawsEnabled: Bool {
        get { gds.enabled }
        set { gds.enabled = newValue }
    }

    /// Diagnostics: (tables built, views drawn on the GPU path, groups the CPU loop still draws per
    /// view, records in the current table).
    public var gpuShadowDrawCensus: (tables: Int, views: Int, fallbackGroups: Int, records: Int) {
        (gds.tablesBuilt, gds.viewsDrawn, gds.lastFallbackGroups, gds.lastRecordCount)
    }

    /// A batch of views written this frame: the slot holding their commands and the table they index.
    struct GPUShadowBatch {
        let slot: GPUShadowFrameSlot
        let table: GPUShadowDrawTable
        /// Groups the CPU loop draws in each of these views.
        var fallback: [Int] { table.fallback }
    }

    /// Step 1: write `specs`' draw commands on the GPU. nil = take the CPU loop for every group
    /// (lever off, no pipelines, a host instance kernel that rewrites the ring, nothing to draw).
    func encodeGPUShadowViews(_ cb: MTLCommandBuffer, _ specs: [GPUShadowViewSpec], spot: Bool) -> GPUShadowBatch? {
        guard gds.enabled, !specs.isEmpty, onEncodeGPUInstances == nil,
              let cull = gpuShadowCullPipeline(), gpuShadowDrawPipeline() != nil,
              let table = currentGPUShadowTable(), table.count > 0, let records = table.records else { return nil }
        let slot = acquireGPUShadowSlot()
        let commands = table.count * specs.count
        guard ensureGPUShadowSlot(slot, commands: commands, views: specs.count),
              let icb = slot.icb, let views = slot.views, let stats = slot.stats,
              let container = slot.container else { return nil }

        // The view block: [views][VP matrices][clock], one shared buffer, read by the kernel and —
        // through the pointers the commands carry — by the vertex stage.
        let viewStride = MemoryLayout<IlluminatoramaGPUShadowView>.stride
        let matBase = viewStride * specs.count
        let clockOff = matBase + MemoryLayout<simd_float4x4>.stride * specs.count
        let base = views.contents()
        base.storeBytes(of: SIMD4<Float>(time, treeWindStrength, treeWindHeading, fieldSinkProgress),
                        toByteOffset: clockOff, as: SIMD4<Float>.self)
        let ring = currentInstanceBuffer
        for (i, s) in specs.enumerated() {
            base.storeBytes(of: s.vp, toByteOffset: matBase + i * MemoryLayout<simd_float4x4>.stride, as: simd_float4x4.self)
            var v = IlluminatoramaGPUShadowView()
            if let c = s.cullVP {
                let p = IlluminatoramaGPUShadowView.planes(c)
                v.planes = (p[0], p[1], p[2], p[3], p[4], p[5])
                v.flags |= IlluminatoramaGPUShadowView.cull
            }
            if s.sun { v.flags |= IlluminatoramaGPUShadowView.sun }
            v.instances = ring.gpuAddress
            v.lightVP = views.gpuAddress + UInt64(matBase + i * MemoryLayout<simd_float4x4>.stride)
            v.clock = views.gpuAddress + UInt64(clockOff)
            v.commandBase = UInt32(i * table.count)
            base.storeBytes(of: v, toByteOffset: i * viewStride, as: IlluminatoramaGPUShadowView.self)
        }
        memset(stats.contents(), 0, 16 * specs.count)
        slot.countsBytes = SIMD2(UInt32(table.count), UInt32(specs.count))

        guard let enc = timedComputeEncoder(cb, spot ? "shadow.spot.gpuDraws" : "shadow.sunCascades.gpuDraws") else { return nil }
        enc.label = "Illuminatorama.shadow.gpuDraws"
        enc.setComputePipelineState(cull)
        enc.setBuffer(records, offset: 0, index: 0)
        enc.setBuffer(views, offset: 0, index: 1)
        enc.setBuffer(container, offset: 0, index: 2)
        enc.setBuffer(stats, offset: 0, index: 3)
        var counts = slot.countsBytes
        enc.setBytes(&counts, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        enc.useResource(icb, usage: .write)
        let w = cull.threadExecutionWidth
        enc.dispatchThreads(MTLSize(width: table.count, height: specs.count, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()

        slot.cb = cb
        slot.keepAlive = [table, records]
        slot.ring = ring
        slot.pending = (spot, specs.count, table.hiddenGroups)
        gds.lastFallbackGroups = table.fallback.count
        gds.lastRecordCount = table.count
        return GPUShadowBatch(slot: slot, table: table)
    }

    /// Step 2: run view `i`'s commands in `enc` (pipeline, depth state, bias and front face are the
    /// encoder's — the ICB inherits them). `cull` tracks the encoder's cull mode, as the CPU loop does.
    func executeGPUShadowView(_ enc: MTLRenderCommandEncoder, _ batch: GPUShadowBatch, view i: Int,
                              cull: inout MTLCullMode) {
        guard let icb = batch.slot.icb, let views = batch.slot.views,
              let pipe = gds.drawPipeline else { return }
        let table = batch.table
        enc.useResources(table.resources, usage: .read, stages: .vertex)
        enc.useResource(views, usage: .read, stages: .vertex)
        if let ring = batch.slot.ring { enc.useResource(ring, usage: .read, stages: .vertex) }
        enc.setRenderPipelineState(pipe)
        let base = i * table.count
        for b in table.buckets where !b.range.isEmpty {
            if b.cull != cull { cull = b.cull; enc.setCullMode(cull) }
            enc.executeCommandsInBuffer(icb, range: (base + b.range.lowerBound) ..< (base + b.range.upperBound))
        }
        gds.viewsDrawn += 1
    }

    /// Fold the census of every GPU-driven batch whose frame has completed into `visibilityCullStats`.
    /// Called before a frame's shadow encode and after a blocking frame's wait, so a blocking caller
    /// reads its own frame's counts.
    func foldGPUShadowStats() {
        for slot in gds.slots {
            guard let p = slot.pending, slot.isFree, let stats = slot.stats else { continue }
            slot.pending = nil
            let s = stats.contents().bindMemory(to: UInt32.self, capacity: 4 * p.views)
            var drawn = 0, culled = p.hidden * p.views, tris = 0
            for v in 0..<p.views {
                drawn += Int(s[v * 4]); culled += Int(s[v * 4 + 1]); tris += Int(s[v * 4 + 2])
            }
            if p.spot {
                visibilityCullStats.spotDrawn += drawn; visibilityCullStats.spotCulled += culled
            } else {
                visibilityCullStats.cascadeDrawn += drawn; visibilityCullStats.cascadeCulled += culled
            }
            visibilityCullStats.trianglesCulled += tris
        }
    }

    // MARK: - Table

    /// The table for this frame's grouping, rebuilt when anything its classification reads moved.
    private func currentGPUShadowTable() -> GPUShadowDrawTable? {
        let key = GPUShadowTableKey(
            groupsVersion: gbs.meshGroupsVersion, meshTableVersion: kMisc.meshTableVersion,
            cullBounds: gbs.cullBoundsKey, cullBoundsCount: gbs.meshGroupCullBounds.count,
            leafCutout: leafCutoutEnabled && shd.shadowPipelineCutout != nil,
            storesFrontFaces: shadowStoresFrontFaces, followsDoubleSided: shadowCullFollowsDoubleSidedForTest,
            hidden: cameraHiddenKinds, noSun: directionalShadowExcludedKinds,
            repacked: fxs.gpuRepackTasks.map(\.kind))
        if let t = gds.table, t.key == key, t.flagsHeld { return t }
        gds.table = buildGPUShadowTable(key)
        return gds.table
    }

    private func buildGPUShadowTable(_ key: GPUShadowTableKey) -> GPUShadowDrawTable {
        gds.tablesBuilt += 1
        let repacked = Set(key.repacked)
        let instStride = MemoryLayout<IlluminatoramaInstance>.stride
        var buckets: [[IlluminatoramaGPUShadowRecord]] = [[], [], []]   // .front, .back, .none
        var bucketMeshes: [[IlluminatoramaMesh]] = [[], [], []]
        var fallback: [Int] = [], hidden = 0
        var resources: [MTLResource] = []
        var seen = Set<ObjectIdentifier>()
        for (gi, group) in gbs.meshGroups.enumerated() {
            guard let mesh = meshes[group.kind], !mesh.tracedOnly else { continue }
            if key.hidden.contains(group.kind) { hidden += 1; continue }
            if mesh.indirectDrawBuffer != nil || mesh.gpuInstanceBuffer != nil || repacked.contains(group.kind)
                || (mesh.leafCutout && key.leafCutout) || mesh.indexCount < 1 {
                fallback.append(gi)
                continue
            }
            var r = IlluminatoramaGPUShadowRecord()
            r.vertices = mesh.vertexBuffer.gpuAddress
            r.indices = mesh.indexBuffer.gpuAddress
            r.instanceOffset = UInt32(instStride * group.start)
            r.indexCount = UInt32(mesh.indexCount)
            r.instanceCount = UInt32(group.count)
            r.triangles = UInt32(clamping: (mesh.indexCount / 3) * group.count)
            if mesh.indexType == .uint32 { r.flags |= IlluminatoramaGPUShadowRecord.uint32Indices }
            if key.noSun.contains(group.kind) { r.flags |= IlluminatoramaGPUShadowRecord.noSunShadow }
            let b = gi < gbs.meshGroupCullBounds.count ? gbs.meshGroupCullBounds[gi] : .unbounded
            if b.isBounded {
                r.flags |= IlluminatoramaGPUShadowRecord.bounded
                r.lo = SIMD4(b.lo, 0); r.hi = SIMD4(b.hi, 0)
            }
            let bucket: Int
            switch shadowCullMode(mesh) { case .front: bucket = 0; case .back: bucket = 1; default: bucket = 2 }
            buckets[bucket].append(r)
            bucketMeshes[bucket].append(mesh)
            if seen.insert(ObjectIdentifier(mesh)).inserted {
                resources.append(mesh.vertexBuffer); resources.append(mesh.indexBuffer)
            }
        }
        let all = buckets.flatMap { $0 }
        let meshesFlat = bucketMeshes.flatMap { $0 }
        var ranges: [(cull: MTLCullMode, range: Range<Int>)] = []
        var at = 0
        for (b, cull) in [MTLCullMode.front, .back, .none].enumerated() {
            ranges.append((cull, at ..< at + buckets[b].count)); at += buckets[b].count
        }
        var buf: MTLBuffer?
        if !all.isEmpty {
            buf = all.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            buf?.label = "Illuminatorama.shadow.gpuDrawRecords"
        }
        return GPUShadowDrawTable(key: key, records: buf, count: all.count, buckets: ranges,
                                  fallback: fallback, hiddenGroups: hidden, resources: resources,
                                  meshes: meshesFlat, flags: meshesFlat.map(GPUShadowDrawTable.flags))
    }

    // MARK: - Slots and pipelines

    private func acquireGPUShadowSlot() -> GPUShadowFrameSlot {
        foldGPUShadowStats()
        if let free = gds.slots.first(where: { $0.isFree }) { return free }
        let s = GPUShadowFrameSlot()
        gds.slots.append(s)
        return s
    }

    private func ensureGPUShadowSlot(_ slot: GPUShadowFrameSlot, commands: Int, views: Int) -> Bool {
        if slot.icb == nil || slot.icbCapacity < commands {
            let d = MTLIndirectCommandBufferDescriptor()
            d.commandTypes = .drawIndexed
            d.inheritPipelineState = true
            d.inheritBuffers = false
            d.maxVertexBufferBindCount = 5
            d.maxFragmentBufferBindCount = 0
            let cap = max(commands, slot.icbCapacity * 2, 1024)
            guard let icb = device.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: cap,
                                                             options: .storageModePrivate),
                  let enc = gds.argumentEncoder,
                  let container = device.makeBuffer(length: enc.encodedLength, options: .storageModeShared) else {
                return false
            }
            icb.label = "Illuminatorama.shadow.gpuDrawICB"
            enc.setArgumentBuffer(container, offset: 0)
            enc.setIndirectCommandBuffer(icb, index: 0)
            slot.icb = icb; slot.icbCapacity = cap; slot.container = container
        }
        let viewBytes = (MemoryLayout<IlluminatoramaGPUShadowView>.stride + MemoryLayout<simd_float4x4>.stride) * views
            + MemoryLayout<SIMD4<Float>>.stride
        if (slot.views?.length ?? 0) < viewBytes {
            slot.views = device.makeBuffer(length: max(viewBytes, 4096), options: .storageModeShared)
            slot.views?.label = "Illuminatorama.shadow.gpuDrawViews"
        }
        if (slot.stats?.length ?? 0) < 16 * views {
            slot.stats = device.makeBuffer(length: max(16 * views, 1024), options: .storageModeShared)
            slot.stats?.label = "Illuminatorama.shadow.gpuDrawStats"
        }
        return slot.views != nil && slot.stats != nil
    }

    private func gpuShadowCullPipeline() -> MTLComputePipelineState? {
        if let p = gds.cullPipeline { return p }
        guard !gds.unavailable, let library = engine.library else { return nil }
        precondition(MemoryLayout<IlluminatoramaGPUShadowRecord>.stride == 80
                     && MemoryLayout<IlluminatoramaGPUShadowView>.stride == 128,
                     "GPU shadow record/view layout drifted from IlluminatoramaGPUDraws.metal")
        do {
            guard let fn = library.makeFunction(name: "illumi_gpu_shadow_draws") else {
                gds.unavailable = true; return nil
            }
            gds.cullPipeline = try device.makeComputePipelineState(function: fn)
            gds.argumentEncoder = fn.makeArgumentEncoder(bufferIndex: 2)
        } catch {
            GPUShadowDrawState.log.error("GPU shadow draws unavailable: \(error.localizedDescription)")
            gds.unavailable = true
        }
        return gds.cullPipeline
    }

    private func gpuShadowDrawPipeline() -> MTLRenderPipelineState? {
        // The depth-only shadow pipeline itself (built indirect-capable) — see `buildShadowPipeline`.
        if gds.drawPipeline == nil, shd.shadowPipeline.supportIndirectCommandBuffers {
            gds.drawPipeline = shd.shadowPipeline
        }
        return gds.drawPipeline
    }
}
