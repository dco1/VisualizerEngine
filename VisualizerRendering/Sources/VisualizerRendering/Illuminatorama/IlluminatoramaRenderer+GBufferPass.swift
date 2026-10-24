import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Dispatch a 1-D compute grid covering `count` threads, threadgroup sized
    /// to the pipeline's execution width.
    func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            count: Int) {
        let tgWidth = max(1, min(pipeline.maxTotalThreadsPerThreadgroup,
                                 pipeline.threadExecutionWidth * 8))
        let groups = (count + tgWidth - 1) / tgWidth
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    }

    func encodeGBufferPass(_ cb: MTLCommandBuffer) {
        // DH-0140 — the lane was decided when this frame's uniforms were written; pick the
        // matching pipelines once. The extended variants exist whenever `extendedGBufferActive`.
        let extended = extendedGBufferActive
        let basePipe: MTLRenderPipelineState = (extended ? gbs.gbufferPipelineExt : nil) ?? gbs.gbufferPipeline
        let prevPipe: MTLRenderPipelineState = (extended ? gbs.gbufferPipelinePrevVertsExt : nil) ?? gbs.gbufferPipelinePrevVerts
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = gbs.gbufferAlbedoMet
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[1].texture = gbs.gbufferNormalRgh
        pass.colorAttachments[1].loadAction = .clear
        pass.colorAttachments[1].storeAction = .store
        pass.colorAttachments[1].clearColor = MTLClearColor(red: 0.5, green: 0.5, blue: 0, alpha: 0)
        pass.colorAttachments[2].texture = gbs.gbufferEmission
        pass.colorAttachments[2].loadAction = .clear
        pass.colorAttachments[2].storeAction = .store
        pass.colorAttachments[2].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        // Phase 2.7 — velocity. Sky pixels (no geometry) stay at zero, which
        // means the TAA resolve treats them as stationary — fine for static
        // skies and visibly slightly smeary for moving clouds.
        pass.colorAttachments[3].texture = pfx.velocityTexture
        pass.colorAttachments[3].loadAction = .clear
        pass.colorAttachments[3].storeAction = .store
        pass.colorAttachments[3].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        // Light-layer target. Cleared to the ALL-BITS default (0xFFFFFFFF) so every
        // sky / no-geometry pixel — and every instance that never sets a layer — is
        // lit by every light exactly as before. The R32Uint clear reads its integer
        // from clearColor.red.
        pass.colorAttachments[4].texture = gbs.gbufferLayer
        pass.colorAttachments[4].loadAction = .clear
        pass.colorAttachments[4].storeAction = .store
        pass.colorAttachments[4].clearColor = MTLClearColor(red: Double(UInt32.max), green: 0, blue: 0, alpha: 0)
        if extended, let material = gbs.gbufferMaterial {
            // Clear to 0 = "not written": impostors and any legacy pipeline that draws into this
            // pass without declaring color(5) leave the clear value, and every reader falls back.
            pass.colorAttachments[5].texture = material
            pass.colorAttachments[5].loadAction = .clear
            pass.colorAttachments[5].storeAction = .store
            pass.colorAttachments[5].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        }
        // DH-1300 — the material-layer slot. Cleared to kMaterialLayerNone (0xFFFFFFFF): sky, and every
        // instance without layer parameters, reads "none" and keeps the legacy material model.
        pass.colorAttachments[7].texture = gbs.gbufferMaterialLayer
        pass.colorAttachments[7].loadAction = .clear
        pass.colorAttachments[7].storeAction = .store
        pass.colorAttachments[7].clearColor = MTLClearColor(red: Double(UInt32.max), green: 0, blue: 0, alpha: 0)
        attachVisBufferTarget(pass)   // Signack S3: the memoryless triangle-id attachment (color 6)
        pass.depthAttachment.texture = gbs.depthTexture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1.0

        // Signack S3 — opted-in meshes are drawn by the visibility buffer at the end of the pass;
        // S4 — their sub-pixel clusters are rasterised in software first, ahead of the pass.
        let cullClip = cullVolume(gbs.gbufferCullVP)   // DH-0534
        let (visDraws, deferred) = collectVisBufferDraws(extended: extended, cullClip: cullClip)
        let soft = encodeSoftRaster(cb, visDraws, extended: extended)

        guard let enc = timedRenderEncoder(cb, pass, "gbuffer") else { return }
        enc.label = "Illuminatorama.gbuffer"
        enc.setRenderPipelineState(basePipe)
        enc.setDepthStencilState(gbs.depthState)
        enc.setFrontFacing(.counterClockwise)
        // Cull mode is a PER-MESH decision here (a two-sided mesh draws `.none`),
        // so it is set inside the draw loop below and deliberately NOT set
        // pass-wide: a pass-wide `.back` is immediately re-issued by the first
        // group, and re-issuing the same value per group is exactly what Metal's
        // validation layer reports as `redundant setCullMode.` /
        // `previous setCullMode was unused.`. Those are severity-2 warnings, but
        // under the debug layer's assert warning mode (Xcode's GPU frame capture
        // / break-on-validation) `_MTLMessageContextEnd…OrAbort` takes the ABORT
        // arm — so a redundant state set is a crash, not a log line. Track the
        // last value and set only on a change, exactly as the shadow passes do.
        // Metal's own default is `.none`, and every drawn group states its cull
        // explicitly, so dropping the pass-wide set changes no pixels.
        var cull: MTLCullMode? = nil

        // Phase 4.0/4.1 — bind the per-material texture atlases once for the
        // whole G-buffer pass. Per-instance slice indices select the slice
        // within each atlas (`< 0` → no sampling, fall back to the per-
        // instance scalar/colour). albedoAtlas is sRGB; nonColorAtlas is
        // linear so metallic/roughness round-trip exactly.
        enc.setFragmentTexture(albedoAtlas.texture, index: 0)
        enc.setFragmentTexture(nonColorAtlas.texture, index: 1)
        // Phase 4.25 (issue #60 item 5) — per-slice UV-scale tables that let the
        // shader tile letterboxed (aspect-preserved) non-square textures at
        // their true aspect. `float2` per slice; indexed by the instance's slice
        // id. Fragment buffer(2) is the instance buffer (set per group below),
        // (4) is unused by the fragment stage — these live at (3)/(5) to match
        // `illumi_fs`'s signature.
        enc.setFragmentBuffer(albedoAtlas.uvScaleBuffer, offset: 0, index: 3)
        enc.setFragmentBuffer(nonColorAtlas.uvScaleBuffer, offset: 0, index: 5)
        // S2.5 — per-slice mean tables (μ for the variance-preserving hex blend). Pass-wide
        // like the UV-scale tables; fragment (7)/(8) are otherwise unused by this pass, and
        // (6) is the superquadric param buffer set per-group below.
        enc.setFragmentBuffer(albedoAtlas.sliceMeanBuffer, offset: 0, index: 7)
        enc.setFragmentBuffer(nonColorAtlas.sliceMeanBuffer, offset: 0, index: 8)
        // Phase 7 — bind the per-frame uniforms to the FRAGMENT stage too (they were
        // already bound to vertex at buffer(1)) so `illumi_fs` can read
        // `antiTilingStrength`. Pass-wide (constant across the draw loop).
        enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 1)
        // Phase 4.12 — instanced draws via the recipe built in
        // `uploadInstances`. One draw call per mesh kind, with the
        // current/previous instance buffer offsets pointed at the start
        // of that group's contiguous slice. The vertex shader's
        // `instances[iid]` indexing then walks the slice naturally.
        // Killed the prior per-instance inner loop — FloatingFlowers+
        // drops from ~1500 draw calls per frame to ~10.
        let instStride = MemoryLayout<IlluminatoramaInstance>.stride
        // Issue #65 — map each deforming GPU-mesh kind to its prev-position side
        // buffer so those groups can draw with the kUsePrevVerts pipeline. Built
        // once per pass (gpuRepackTasks is tiny — one entry per registered mesh).
        var prevPosByKind: [MeshKind: MTLBuffer] = [:]
        for task in fxs.gpuRepackTasks where !task.direct { prevPosByKind[task.kind] = task.prevPositionBuffer }
        // Track which pipeline is bound to avoid redundant state switches when the
        // groups are all ordinary meshes (the common case).
        var prevVertsBound = false
        var interiorBound = false
        var cutoutBound = false   // DH-1487
        var substrateBiased = false
        for (gi, group) in gbs.meshGroups.enumerated() {
            if deferred.contains(gi) { continue }   // the visibility buffer draws it (culled above)
            // Superquadric impostor box kinds are drawn by the impostor pipeline
            // below; RT-proxy kinds exist only in the TLAS; shadow-only kinds are
            // lighting occluders that draw solely into the shadow maps. All skipped here.
            if gbs.impostorMeshKinds.contains(group.kind) ||
               rts.rtProxyMeshKinds.contains(group.kind) ||
               shadowOnlyMeshKinds.contains(group.kind) { continue }
            guard let mesh = meshes[group.kind] else { continue }
            if shouldCull(gi, group, mesh, cullClip, drawn: \.gbufferDrawn, culled: \.gbufferCulled) { continue }
            // Two-sided meshes (open / dynamic MC fluid surfaces) render cull
            // `.none` so they don't go hollow when their back side faces the
            // camera; the fragment shader flips the normal for back faces.
            let want: MTLCullMode = mesh.doubleSided ? .none : .back
            if cull != want { cull = want; enc.setCullMode(want) }
            // The substrate takes a depth-buffer bias so a millimetre of lift on top of it keeps
            // winning at any distance (see `substrateMeshKinds`). Tracked like `cull` — one
            // state change per run of groups, not per group.
            let wantBias = substrateMeshKinds.contains(group.kind)
            if wantBias != substrateBiased {
                substrateBiased = wantBias
                enc.setDepthBias(wantBias ? substrateDepthBias : 0, slopeScale: 0, clamp: 0)
            }
            let off = instStride * group.start
            // Issue #65 — deforming GPU meshes draw with the kUsePrevVerts pipeline
            // and bind their per-vertex prev positions at buffer(5); every other
            // group uses the base pipeline (which declares no buffer(5)).
            let prevPos = prevPosByKind[group.kind]
            // Interior-mapped kinds (opt-in): their own pipeline + params at fragment buffer(9).
            if let imap = interiorMappedMeshKinds[group.kind], prevPos == nil,
               let ip = interiorGBufferPipeline(extended: extended) {
                if !interiorBound { enc.setRenderPipelineState(ip) }
                interiorBound = true
                prevVertsBound = false
                cutoutBound = false
                var gpu = imap.gpu
                enc.setFragmentBytes(&gpu, length: MemoryLayout<IlluminatoramaInteriorMapping.GPU>.stride, index: 9)
            } else if prevPos != nil {
                if !prevVertsBound || interiorBound || cutoutBound { enc.setRenderPipelineState(prevPipe) }
                prevVertsBound = true
                interiorBound = false
                cutoutBound = false
            } else if mesh.leafCutout, leafCutoutEnabled, let cp = leafCutoutGBufferPipeline(extended: extended) {
                if !cutoutBound { enc.setRenderPipelineState(cp) }   // DH-1487
                cutoutBound = true
                prevVertsBound = false
                interiorBound = false
            } else if prevVertsBound || interiorBound || cutoutBound {
                enc.setRenderPipelineState(basePipe)
                prevVertsBound = false
                interiorBound = false
                cutoutBound = false
            }
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
            let inst = mesh.instanceBinding(ring: currentInstanceBuffer, groupOffset: off)
            enc.setVertexBuffer(inst.buffer, offset: inst.offset, index: 2)
            enc.setFragmentBuffer(inst.buffer, offset: inst.offset, index: 2)
            // DH-1300 — the group's first record in the ring, so the material-layer slot is global (the
            // lighting / RT kernels read the ring from 0); a GPU-compacted batch is not in the ring.
            var layerSlotBase: UInt32 = inst.buffer === currentInstanceBuffer ? UInt32(inst.offset / instStride) : UInt32.max
            enc.setFragmentBytes(&layerSlotBase, length: 4, index: 10)
            // Phase 2.7 — previous-frame instances at buffer(4) for the
            // motion-vector reconstruction. Same per-group offset as
            // current — the ping-pong buffers share the grouped layout
            // because the host's `instances` order is stable across
            // frames for a static scene topology.
            let prev = mesh.instanceBinding(ring: previousInstanceBuffer, groupOffset: off)   // S6: same records
            enc.setVertexBuffer(prev.buffer, offset: prev.offset, index: 4)
            // Issue #65 — per-vertex previous positions for the deforming-mesh
            // motion vector. Only the kUsePrevVerts pipeline reads buffer(5).
            if let prevPos { enc.setVertexBuffer(prevPos, offset: 0, index: 5) }
            enc.drawMesh(mesh, instanceCount: group.count, withSoft: !vbs.soft.dropForTest)
        }

        // ── Perfect analytic superquadric impostors ───────────────────────────
        // Drawn into the SAME G-buffer pass with the impostor pipeline: each
        // instance's [-1,1] bounding-box mesh is rasterized (BACK faces — cull
        // `.front` — so the primitive survives the camera entering the box), and
        // the fragment ray-traces the analytic surface, writing the G-buffer +
        // analytic depth + analytic motion vectors. The param buffer is bound at
        // the same per-group offset as the instance buffer (grouped-order aligned).
        if substrateBiased { substrateBiased = false; enc.setDepthBias(0, slopeScale: 0, clamp: 0) }

        if let sqPipe = gbs.superquadricImpostorPipeline, !gbs.impostorMeshKinds.isEmpty {
            enc.setRenderPipelineState(sqPipe)
            if cull != .front { cull = .front; enc.setCullMode(.front) }
            let pStride = MemoryLayout<IlluminatoramaInstance.SuperquadricParam>.stride
            for group in gbs.meshGroups where gbs.impostorMeshKinds.contains(group.kind) {
                guard let mesh = meshes[group.kind] else { continue }
                let off  = instStride * group.start
                let poff = pStride * group.start
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
                enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 1)
                enc.setFragmentBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.setFragmentBuffer(previousInstanceBuffer, offset: off, index: 4)
                enc.setFragmentBuffer(superquadricParamBuffer, offset: poff, index: 6)
                var layerSlotBase = UInt32(group.start)   // DH-1300 — the group's first ring record
                enc.setFragmentBytes(&layerSlotBase, length: 4, index: 10)
                enc.drawMesh(mesh, instanceCount: group.count)
            }
        }
        // Signack S3 — the deferred meshes, last: ids against everything above, then the resolve.
        encodeVisBuffer(enc, visDraws, extended: extended, cull: cull, soft: soft)
        enc.endEncoding()
    }

    /// The groups the visibility buffer draws this frame (Signack S3), decided BEFORE the pass so the
    /// software rasterizer (S4) can run ahead of it. Culling is decided here for them (and counted
    /// once); `deferred` holds every group index this claimed, culled or not.
    private func collectVisBufferDraws(extended: Bool, cullClip: IlluminatoramaClipVolume?)
        -> ([VisBufferDeferredDraw], Set<Int>) {
        guard visBufferActive(extended: extended) else { return ([], []) }
        var draws: [VisBufferDeferredDraw] = [], claimed = Set<Int>()
        for (gi, group) in gbs.meshGroups.enumerated() where draws.count < VisBufferState.maxDraws {
            if gbs.impostorMeshKinds.contains(group.kind) || rts.rtProxyMeshKinds.contains(group.kind)
                || shadowOnlyMeshKinds.contains(group.kind) { continue }
            guard let mesh = meshes[group.kind], defersToVisBuffer(mesh, kind: group.kind, count: group.count) else { continue }
            claimed.insert(gi)
            if shouldCull(gi, group, mesh, cullClip, drawn: \.gbufferDrawn, culled: \.gbufferCulled) { continue }
            draws.append(VisBufferDeferredDraw(group: group, mesh: mesh,
                bounds: gi < gbs.meshGroupCullBounds.count ? gbs.meshGroupCullBounds[gi] : .unbounded))
        }
        return (draws, claimed)
    }

    func dispatch(_ enc: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          width: Int, height: Int,
                          maxThreadsOverride: Int? = nil) {
        let tgw = pipeline.threadExecutionWidth
        // Default = the full threadgroup (max occupancy by thread count). For
        // register-heavy kernels (RT) a SMALLER group can run faster — more
        // threadgroups resident per core. `maxThreadsOverride` lets a specific
        // pass cap the total threads (#60 task 4 occupancy tuning).
        let cap = pipeline.maxTotalThreadsPerThreadgroup
        let total = max(tgw, min(cap, maxThreadsOverride ?? cap))
        let tgh = max(1, total / tgw)
        let threadgroup = MTLSize(width: tgw, height: tgh, depth: 1)
        let groups = MTLSize(
            width:  (width  + tgw - 1) / tgw,
            height: (height + tgh - 1) / tgh,
            depth: 1
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroup)
    }

    /// 3D dispatch covering (faceSize × faceSize × 6) for cubemap-baking
    /// kernels that take `[[thread_position_in_grid]]` as `uint3` and use
    /// `gid.z` as the face index. Threadgroups are 8×8×1 — depth=1 because
    /// faces don't share state and that keeps the threadgroup tile small
    /// enough to fit on every device.
    func dispatchCubeFaces(_ enc: MTLComputeCommandEncoder,
                                   pipeline: MTLComputePipelineState,
                                   faceSize: Int) {
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(
            width:  (faceSize + tg.width - 1)  / tg.width,
            height: (faceSize + tg.height - 1) / tg.height,
            depth:  6
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }
}
