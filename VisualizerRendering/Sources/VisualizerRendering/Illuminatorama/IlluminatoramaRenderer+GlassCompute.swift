import Foundation
import Metal
import simd

// MARK: - RT glass on a compute pass (Daydream DH-1625)
//
// Measured on 4000 Sunset (M1, software ray traversal, settled 1920×1200): the RT glass FRAGMENT pass
// cost ~25 ms on ~27 k glass fragments — ~350 ns a reflection ray, where the compute TLAS lighting pass
// spends a few ns a ray. The fragment stage is a poor home for a deep ray-query walk. So by default the
// raster only finds the nearest pane per pixel (`illumi_glass_surface_fs` into two small targets, depth
// written into a copy of the scene depth) and `illumi_glass_rt_cs` runs the SAME shading body
// (`glassShadePixel`) over those pixels. The defocus trace (opt-in, VZ-0197) keeps the fragment path.

extension IlluminatoramaRenderer {
    /// Where a glass pass's RT bindings go: the fragment stage or a compute encoder. One binding list,
    /// two sinks — the fragment and compute paths cannot bind different things.
    struct GlassBinder {
        let setBuffer: (MTLBuffer?, Int, Int) -> Void
        let setTexture: (MTLTexture?, Int) -> Void
        let accel: (MTLAccelerationStructure, Int) -> Void
        let use: (MTLResource) -> Void

        func buffer(_ b: MTLBuffer?, offset: Int, index: Int) { setBuffer(b, offset, index) }
        func texture(_ t: MTLTexture?, index: Int) { setTexture(t, index) }

        static func fragment(_ enc: MTLRenderCommandEncoder) -> GlassBinder {
            GlassBinder(setBuffer: { enc.setFragmentBuffer($0, offset: $1, index: $2) },
                        setTexture: { enc.setFragmentTexture($0, index: $1) },
                        accel: { enc.setFragmentAccelerationStructure($0, bufferIndex: $1) },
                        use: { enc.useResource($0, usage: .read, stages: .fragment) })
        }
        static func compute(_ enc: MTLComputeCommandEncoder) -> GlassBinder {
            GlassBinder(setBuffer: { enc.setBuffer($0, offset: $1, index: $2) },
                        setTexture: { enc.setTexture($0, index: $1) },
                        accel: { enc.setAccelerationStructure($0, bufferIndex: $1) },
                        use: { enc.useResource($0, usage: .read) })
        }
    }

    /// The RT glass shader's scene bindings (TLAS, instance data, surface cache, lights, atlas…). False when
    /// the TLAS side is not live — the caller then has nothing to trace.
    @discardableResult
    func bindGlassRTResources(_ b: GlassBinder, cacheOn: Bool, statsBinding: MTLBuffer?) -> Bool {
        guard let tlas = rts.rtTLAS, let instData = rts.rtInstanceDataBuffer,
              let objN = rts.rtObjNormalBuffer, let gData = gls.rtGlassDataBuffer else { return false }
        b.accel(tlas, 0)
        b.buffer(instData, offset: 0, index: 3)
        b.buffer(objN, offset: 0, index: 4)
        b.buffer(gData, offset: 0, index: 5)
        // Surface-cache buffers (dummies keep the bindings valid when off);
        // optional-ternary mirrors the TLAS lighting pass — no force-unwrap.
        let dummy: MTLBuffer = instData
        b.buffer(cacheOn ? scs.surfTriCardBuffer : dummy, offset: 0, index: 6)
        b.buffer(cacheOn ? scs.surfTriUVaBuffer : dummy, offset: 0, index: 7)
        b.buffer(cacheOn ? scs.surfTriUVcBuffer : dummy, offset: 0, index: 8)
        b.buffer(cacheOn ? rts.rtSoupTriBaseBuffer : dummy, offset: 0, index: 9)
        b.buffer(cacheOn ? scs.surfCardRectBuffer : dummy, offset: 0, index: 10)
        b.buffer(cacheOn ? scs.surfCardBuffer : (surfCardDummyBuffer ?? dummy), offset: 0, index: 11)
        b.texture(cacheOn ? surfConsumerAtlas : (equirectSky ?? ibls.dummySkyTexture), index: 1)
        // Through-glass world parity: re-shaded opaque hits sample the SAME
        // cosine-convolved irradiance cube the deferred lighting pass uses for
        // its diffuse IBL, so transmitted scenery gets the sky fill (not just
        // the flat ambient supplement) and matches the world beside the pane.
        b.texture(ibls.irradianceCube, index: 3)
        // Through-glass TEXTURE parity: the per-triangle mesh UVs + the albedo
        // atlas (and its per-slice letterbox table), so a re-shaded hit samples
        // the SAME texel the G-buffer would instead of the instance's mean
        // albedo. Without these the world behind a pane is flat slabs.
        b.buffer(rts.rtObjUVBuffer ?? instData, offset: 0, index: 12)
        b.buffer(albedoAtlas.uvScaleBuffer, offset: 0, index: 13)
        b.texture(albedoAtlas.texture, index: 4)
        // Local lights: the deferred kernel accumulates point + spot; the glass
        // pass used to have sun-only, so a lamp-lit room went black through a
        // window. Same ring buffers the lighting kernel reads this frame.
        b.buffer(pointLightBuffer, offset: 0, index: 14)
        b.buffer(spotLightBuffer, offset: 0, index: 15)
        b.buffer(shd.spotProfileBuffer, offset: 0, index: 18)   // DH-0540 — the spot-profile rows
        b.buffer(cloudShadowBinding, offset: 0, index: 19)      // DH-1462 — cloud sun transmittance
        // Smooth refraction: glass hits interpolate these corner normals (a dummy with
        // every row's base 0 means "face normal" — see `rtGlassCornerNormalBuffer`).
        b.buffer(gls.rtGlassCornerNormalBuffer ?? instData, offset: 0, index: 16)
        b.buffer(statsBinding, offset: 0, index: 21)   // DH-1625 — work census
        // DH-0718 — area lights at the refracted hit (buffer 17); dummy when none.
        b.buffer(secondaryAreaLightCount > 0 ? areaLightBuffer : instData,
                              offset: 0, index: 17)
        // The TLAS references the BLASes which reference mesh buffers — all
        // must be resident for the intersector (fragment or compute).
        for blas in rts.rtBLASList { b.use(blas) }
        for buf in rts.rtResidentBuffers { b.use(buf) }
        return true
    }

    /// Lever for the A/B (DH-1625): false sends RT glass back through the fragment pass. Default on.
    public var rtGlassComputeEnabled: Bool {
        get { gls.computeEnabled }
        set { gls.computeEnabled = newValue }
    }

    /// The compute path's two surface targets + its depth copy, at the composite's size.
    func ensureGlassSurfaceTargets() -> (pos: MTLTexture, normal: MTLTexture, depth: MTLTexture)? {
        let w = pfx.hdrCompositeTexture.width, h = pfx.hdrCompositeTexture.height
        if gls.surfacePos?.width != w || gls.surfacePos?.height != h {
            func make(_ f: MTLPixelFormat, _ usage: MTLTextureUsage, _ label: String) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: w, height: h, mipmapped: false)
                d.usage = usage
                d.storageMode = .private
                let t = device.makeTexture(descriptor: d)
                t?.label = label
                return t
            }
            // Full float: the kernel must start its trace from exactly the point the fragment path did.
            gls.surfacePos = make(.rgba32Float, [.renderTarget, .shaderRead], "Illuminatorama.glass.surfacePos")
            gls.surfaceNormal = make(.rgba32Float, [.renderTarget, .shaderRead], "Illuminatorama.glass.surfaceNormal")
            gls.surfaceDepth = make(.depth32Float, [.renderTarget], "Illuminatorama.glass.surfaceDepth")
        }
        guard let p = gls.surfacePos, let n = gls.surfaceNormal, let d = gls.surfaceDepth else { return nil }
        return (p, n, d)
    }

    /// RT glass as raster-then-compute. False ⇒ nothing was encoded (pipelines or targets missing) and the
    /// caller falls back to the fragment pass.
    func encodeGlassCompute(_ cb: MTLCommandBuffer, groups: [(kind: MeshKind, insts: [IlluminatoramaGlassInstance])],
                            offsets: [Int], cacheOn: Bool, statsBinding: MTLBuffer?) -> Bool {
        guard let targets = ensureGlassSurfaceTargets(), let library = engine.library else { return false }
        if gls.surfacePipeline == nil {
            gls.surfacePipeline = Self.makeGlassSurfacePipeline(device: device, library: library)
            let ds = MTLDepthStencilDescriptor()
            ds.depthCompareFunction = .lessEqual
            ds.isDepthWriteEnabled = true                      // nearest pane wins
            gls.surfaceDepthState = device.makeDepthStencilState(descriptor: ds)
        }
        let cv = MTLFunctionConstantValues()
        var stats = gls.statsEnabled
        cv.setConstantValue(&stats, type: .bool, index: Self.glassStatsFunctionConstantIndex)
        guard let surfacePipe = gls.surfacePipeline, let surfaceDS = gls.surfaceDepthState,
              let kernel = engine.pipelineCache.pipelineState(name: "illumi_glass_rt_cs", device: device,
                                                              constants: cv, variantKey: stats ? "stats" : "ship")
        else { return false }

        // 1 — the scene depth, copied: the surface pass writes into it, and later passes read the original.
        guard let blit = cb.makeBlitCommandEncoder() else { return false }
        blit.label = "Illuminatorama.glass.depthCopy"
        blit.copy(from: gbs.depthTexture, to: targets.depth)
        blit.endEncoding()

        // 2 — the nearest pane per pixel. Back faces culled exactly as the RT fragment pass culls them.
        let pass = MTLRenderPassDescriptor()
        for (i, t) in [targets.pos, targets.normal].enumerated() {
            pass.colorAttachments[i].texture = t
            pass.colorAttachments[i].loadAction = .clear      // w = 0: no glass
            pass.colorAttachments[i].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[i].storeAction = .store
        }
        pass.depthAttachment.texture = targets.depth
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .dontCare
        guard let enc = timedRenderEncoder(cb, pass, "glass.surface") else { return false }
        enc.label = "Illuminatorama.glass.surface"
        enc.setRenderPipelineState(surfacePipe)
        enc.setDepthStencilState(surfaceDS)
        enc.setCullMode(.back)
        enc.setFrontFacing(.counterClockwise)
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
        enc.setVertexBuffer(glassRTUniformBuffer, offset: 0, index: 3)
        let stride = MemoryLayout<IlluminatoramaGlassInstance>.stride
        for (i, g) in groups.enumerated() {
            guard let mesh = meshes[g.kind] else { continue }
            var base = UInt32(offsets[i])
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(glassInstanceBuffer, offset: stride * offsets[i], index: 2)
            enc.setFragmentBytes(&base, length: MemoryLayout<UInt32>.size, index: 4)
            enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                      indexType: mesh.indexType, indexBuffer: mesh.indexBuffer,
                                      indexBufferOffset: 0, instanceCount: g.insts.count)
        }
        enc.endEncoding()

        // 3 — trace every glass pixel. Same bindings as the fragment pass, plus the targets.
        guard let cenc = timedComputeEncoder(cb, "glass") else { return false }
        cenc.label = "Illuminatorama.glass.rt.compute"
        cenc.setComputePipelineState(kernel)
        cenc.setBuffer(glassRTUniformBuffer, offset: 0, index: 1)
        cenc.setBuffer(glassInstanceBuffer, offset: 0, index: 2)
        let useBackdrop = rtGlassScreenSpaceTransmission
        cenc.setTexture(equirectSky ?? ibls.dummySkyTexture, index: 0)
        cenc.setTexture((useBackdrop ? gls.glassBackdropTexture : nil) ?? ibls.dummySkyTexture, index: 2)
        guard bindGlassRTResources(.compute(cenc), cacheOn: cacheOn, statsBinding: statsBinding) else {
            cenc.endEncoding(); return false
        }
        cenc.setTexture(targets.pos, index: 6)
        cenc.setTexture(targets.normal, index: 7)
        cenc.setTexture(pfx.hdrCompositeTexture, index: 8)
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        cenc.dispatchThreads(MTLSize(width: targets.pos.width, height: targets.pos.height, depth: 1),
                             threadsPerThreadgroup: tg)
        cenc.endEncoding()
        return true
    }

    /// The nearest-surface raster: the glass vertex stage + `illumi_glass_surface_fs`, two float targets.
    static func makeGlassSurfacePipeline(device: MTLDevice, library: MTLLibrary) -> MTLRenderPipelineState? {
        guard let vs = library.makeFunction(name: "illumi_glass_rt_vs"),
              let fs = library.makeFunction(name: "illumi_glass_surface_fs") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "Illuminatorama.glass.surface"
        d.vertexFunction = vs
        d.fragmentFunction = fs
        d.colorAttachments[0].pixelFormat = .rgba32Float
        d.colorAttachments[1].pixelFormat = .rgba32Float
        d.depthAttachmentPixelFormat = .depth32Float
        return try? device.makeRenderPipelineState(descriptor: d)
    }
}
