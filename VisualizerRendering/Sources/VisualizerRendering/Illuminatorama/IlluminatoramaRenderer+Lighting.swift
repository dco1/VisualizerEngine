import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {

    // MARK: - Lighting kernel specialization (WWDC23 #10127)

    /// Builds the `(variantKey, constants)` pair for `illumi_lighting`'s
    /// function-constant specialization from the host feature flags. The booleans
    /// bind at indices 0–5 matching the `kLighting*` declarations in
    /// Illuminatorama.metal; the key is the 6-bit string that disambiguates the
    /// variant in the pipeline cache. These MUST agree with the `frame.*Enabled`
    /// uniforms written in `updateFrameUniforms` — both derive from the same Bools.
    static func lightingFeatureConstants(ibl: Bool, shadow: Bool, dfg: Bool,
                                         rtSunShadow: Bool = false,
                                         aov: Bool = false)
        -> (key: String, constants: MTLFunctionConstantValues) {
        // index 5 — S4.1 ray-traced soft sun shadows. The caller passes the fully
        // derived bit (`rtSunShadowLightingActive`), never the raw flag: the RT
        // variant's TLAS argument only exists when this is true, so selecting it
        // without a live TLAS would be an unbindable pipeline.
        var flags = [ibl, shadow, dfg]
        let cv = MTLFunctionConstantValues()
        flags.withUnsafeMutableBufferPointer { buf in
            for i in 0..<buf.count {
                cv.setConstantValue(buf.baseAddress! + i, type: .bool, index: i)
            }
        }
        var rtSunBit = rtSunShadow
        cv.setConstantValue(&rtSunBit, type: .bool, index: 5)
        var key = (flags + [rtSunShadow]).map { $0 ? "1" : "0" }.joined()
        // Daydream DH-1304 — the AOV capture variant (index 8): set only while capturing, so every
        // other key (and every other host) keeps the constant undefined.
        if aov {
            var on = true
            cv.setConstantValue(&on, type: .bool, index: lightingAOVFunctionConstantIndex)
            key += "-aov"
        }
        return (key, cv)
    }

    /// The fully derived S4.1 pipeline bit: soft sun shadows requested AND a TLAS
    /// is genuinely live to trace. `rtTLASActive` is set by `updateRTAccel`, which
    /// runs AFTER the lighting pass in the frame graph — so the kernel traces the
    /// PREVIOUS frame's TLAS, and on the very first frame (or any frame the RT
    /// hang-guard trips) this is false and lighting falls back to the shadow-map
    /// variant instead of selecting a pipeline whose accel argument cannot bind.
    private var rtSunShadowLightingActive: Bool {
        rtSunSoftShadowsEnabled && rts.rtTLASActive && rts.rtTLAS != nil
    }

    /// The specialized `illumi_lighting` pipeline for the current feature flags.
    /// The shared cache compiles each flag combination once and returns the
    /// memoised state thereafter; this method additionally short-circuits the
    /// common no-change frame before touching the cache.
    ///
    /// WWDC23 #10127 lever 2 (async compilation): when a flag toggle first needs
    /// a not-yet-compiled variant, we compile it in the BACKGROUND (no frame
    /// hitch) and run the init-time uber-variant (`lightingPipeline`) meanwhile —
    /// it renders identical pixels, just without that combination's dead-code
    /// elimination. The fallback is deliberately NOT memoised, so once the
    /// background compile lands the next frame picks up the specialized variant.
    private func currentLightingPipeline() -> MTLComputePipelineState {
        let rtSun = rtSunShadowLightingActive
        let aov = aovs.aovActiveThisFrame
        let bits: UInt8 = (iblEnabled     ? 1  : 0)
                        | (shadowsEnabled ? 2  : 0)
                        | (dfgLUTEnabled  ? 4  : 0)
                        | (rtSun          ? 32 : 0)
                        | (aov            ? 64 : 0)
        if bits == gbs.lastLightingFlags, let p = gbs.lastLightingPipeline { return p }

        let (key, cv) = Self.lightingFeatureConstants(
            ibl: iblEnabled, shadow: shadowsEnabled, dfg: dfgLUTEnabled,
            rtSunShadow: rtSun, aov: aov)
        // Headless/settled lane (no runloop to land the async completion — a
        // @MainActor render loop starves the MainActor hop): compile the variant
        // BLOCKING so the frame actually runs the specialized path instead of the
        // init-time uber fallback. DH-0552.
        // DH-1304 — a capture frame compiles BLOCKING too: the uber fallback writes no AOVs.
        if blockingLightingCompile || aov {
            if let built = engine.pipelineCache.pipelineState(
                name: "illumi_lighting", device: device, constants: cv, variantKey: key) {
                gbs.lastLightingFlags = bits
                gbs.lastLightingPipeline = built
                return built
            }
            return gbs.lightingPipeline
        }
        // Non-blocking: ready variant → memoise + use it; still compiling → run
        // the uber-variant this frame and re-check next frame (no memo).
        if let ready = engine.pipelineCache.pipelineStateAsync(
            name: "illumi_lighting", device: device, constants: cv, variantKey: key) {
            gbs.lastLightingFlags = bits
            gbs.lastLightingPipeline = ready
            return ready
        }
        // S4.1 — the RT-sun variant compiles BLOCKING instead of falling back. The
        // bit is only ever set in a settled/photo lane (no vsync budget), and the
        // fallback there is not benign: the still's accumulator would blend
        // shadow-map frames into the RT penumbra while the background compile
        // lands. The variant is deliberately NOT in the init pre-warm either —
        // every Visualizer host would pay its compile for a feature none of them
        // set. (Not memoised on failure; a nil build falls through to the uber
        // variant exactly as before.)
        if rtSun, let built = engine.pipelineCache.pipelineState(
            name: "illumi_lighting", device: device, constants: cv, variantKey: key) {
            gbs.lastLightingFlags = bits
            gbs.lastLightingPipeline = built
            return built
        }
        return gbs.lightingPipeline
    }

    /// Compute encoder that records this pass's GPU time when the pass timer is
    /// enabled (env-gated; #60 task 4 / #6), else a plain encoder. The `label`
    /// keys the per-pass breakdown.
    func timedComputeEncoder(_ cb: MTLCommandBuffer, _ label: String) -> MTLComputeCommandEncoder? {
        if let pd = gbs.passTimer.descriptor(label: label) {
            return cb.makeComputeCommandEncoder(descriptor: pd)
        }
        return cb.makeComputeCommandEncoder()
    }

    /// Raster counterpart of `timedComputeEncoder` — brackets the render pass with
    /// the same GPU timestamp pair so the G-buffer / shadow / glass / tonemap
    /// passes appear in the per-pass breakdown instead of vanishing into an
    /// untimed remainder. No-op (plain encoder) when the pass timer is off.
    func timedRenderEncoder(_ cb: MTLCommandBuffer,
                                    _ pass: MTLRenderPassDescriptor,
                                    _ label: String) -> MTLRenderCommandEncoder? {
        cb.makeRenderCommandEncoder(descriptor: gbs.passTimer.attach(to: pass, label: label))
    }

    func encodeLightingPass(_ cb: MTLCommandBuffer) {
        // Defocus shading rate (opt-in): the tile map first, then the coarse + skip variants below.
        let rate = lightingShadingRateRequested ? encodeLightingRateTiles(cb) : nil
        guard let enc = timedComputeEncoder(cb, "lighting") else { return }
        enc.label = "Illuminatorama.lighting"
        let pipe = currentLightingPipeline()
        enc.setComputePipelineState(pipe)
        enc.setTexture(gbs.gbufferAlbedoMet, index: 0)
        enc.setTexture(gbs.gbufferNormalRgh, index: 1)
        enc.setTexture(gbs.gbufferEmission, index: 2)
        enc.setTexture(gbs.depthTexture, index: 3)
        enc.setTexture(gbs.hdrTexture, index: 4)
        enc.setTexture(aoSourceTexture, index: 5)  // denoised AO (or raw when off)
        enc.setTexture(equirectSky ?? ibls.dummySkyTexture, index: 6)
        enc.setTexture(ibls.irradianceCube, index: 7)
        enc.setTexture(ibls.prefilteredCube, index: 8)
        enc.setTexture(shd.shadowMap, index: 9)
        enc.setTexture(ibls.dfgLUT, index: 10)
        // Phase 4.10 — per-spot shadow atlas. Always bound (the kernel
        // gates per-spot on `spotShadowSliceIndex >= 0`, which is set by
        // `updateSpotShadows` only when `spotShadowsEnabled` is on AND
        // the spot is within atlas capacity).
        enc.setTexture(shd.spotShadowAtlas, index: 13)
        // Point-light cube depth atlas (6 slices/light). When the feature is off
        // its `pointShadowAtlas` is nil → bind the always-allocated spot atlas as a
        // same-type (depth2d_array) placeholder; the kernel never samples it because
        // every point light's `shadowCubeIndex` is -1. When on, real cube depth.
        enc.setTexture(shd.pointShadowAtlas ?? shd.spotShadowAtlas, index: 19)
        enc.setTexture(ibls.ltcMatTexture, index: 16)
        enc.setTexture(ibls.ltcMagTexture, index: 17)
        // Light-layer G-buffer target — per-fragment layer bitfield the kernel
        // masks each light against. When no scene sets layers this reads all-bits
        // everywhere, so every light passes the mask (no behaviour change).
        enc.setTexture(gbs.gbufferLayer, index: 18)
        // DH-1300 — the material-layer slot target + the instance buffer it indexes (the kernel reads
        // the instance's layer parameters by slot; IlluminatoramaMaterialLayers.h).
        enc.setTexture(gbs.gbufferMaterialLayer, index: 26)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 2)
        // Issue #65 — SSS diffuse side buffer. Always bound (the kernel only writes
        // it when frame.sssStrength > 0, so non-SSS scenes leave it idle).
        // (index 20: 18/19 went to gLayer/pointShadowAtlas in the merge.)
        enc.setTexture(ssrs.sssDiffuseTexture, index: 20)
        // DH-0140 — the kernel reads this only when frame.extendedGBuffer != 0; any texture
        // satisfies the binding otherwise.
        enc.setTexture((extendedGBufferActive ? gbs.gbufferMaterial : nil) ?? gbs.gbufferEmission, index: 21)
        // DH-0896 — the specular-IBL hand-off to the RT reflection pass (see `specIBLTexture`).
        let specTarget = specIBLHandoffTarget()
        ibls.specIBLWrittenThisFrame = specTarget != nil
        enc.setTexture(specTarget ?? specIBLDummyTexture, index: 22)
        let diffSkyTarget = diffSkyHandoffTarget()
        ibls.diffSkyWrittenThisFrame = diffSkyTarget != nil
        enc.setTexture(diffSkyTarget ?? diffSkyDummyTexture, index: 23)
        // DH-1304 — the AOV capture (texture 25 + buffer 10 exist only in the `-aov` variants).
        if aovs.aovActiveThisFrame, let aovTex = aovs.aovTexture {
            enc.setTexture(aovTex, index: 25)
            var au = AOVUniforms(blend: aovs.aovFrameBlend, handoff: diffSkyTarget != nil ? 1 : 0)
            enc.setBytes(&au, length: MemoryLayout<AOVUniforms>.stride, index: 10)
        }
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        enc.setBuffer(pointLightBuffer, offset: 0, index: 1)
        enc.setBuffer(spotLightBuffer, offset: 0, index: 3)
        enc.setBuffer(areaLightBuffer, offset: 0, index: 4)
        enc.setBuffer(extraDirectionalBuffer, offset: 0, index: 5)
        enc.setBuffer(shd.spotProfileBuffer, offset: 0, index: 9)   // DH-0540 — the spot-profile rows
        enc.setBuffer(cloudShadowBinding, offset: 0, index: 11)     // DH-1462 — cloud sun transmittance
        // Point-light cube-face view-projection matrices (6 per shadowed light).
        // Placeholder-bound to `pointLightBuffer` when the feature is off / not yet
        // allocated — never read, because the kernel only indexes it for lights with
        // `shadowCubeIndex >= 0`, and those exist only when the feature is on.
        enc.setBuffer(pointShadowFaceBuffer ?? pointLightBuffer, offset: 0, index: 6)
        // Interior daylight apertures (S3.5 Stage D). Always bound — the kernel gates
        // on the count/strength packed in frame.interiorIrrDown.w / interiorIrrSide.w,
        // both 0 for every scene that never opts in.
        enc.setBuffer(interiorApertureBuffer, offset: 0, index: 7)
        // S4.1 — the TLAS for ray-traced soft sun shadows. Bound exactly when the
        // pipeline bit is set (the accel argument only exists in that variant; if
        // the specialized variant is still compiling and the uber fallback runs
        // this frame, the extra binding is simply unread).
        if rtSunShadowLightingActive, let tlas = rts.rtTLAS {
            enc.setAccelerationStructure(tlas, bufferIndex: 8)
        }
        if let rate, let variants = lightingShadingRatePipelines() {
            // Same bindings; the tile map at texture(24). Coarse quads first (a thread per quad),
            // then every other pixel — the two variants' pixel sets partition the frame.
            enc.setTexture(rate, index: 24)
            enc.setComputePipelineState(variants.coarse)
            dispatch(enc, pipeline: variants.coarse, width: (width + 1) / 2, height: (height + 1) / 2)
            enc.setComputePipelineState(variants.skip)
            dispatch(enc, pipeline: variants.skip, width: width, height: height)
        } else {
            dispatch(enc, pipeline: pipe, width: width, height: height)
        }
        enc.endEncoding()
    }

    /// Whether this frame asks for the defocus shading rate (the host's knob, with a live lens).
    var lightingShadingRateRequested: Bool {
        lightingDefocusShadingMinCoC > 0 && dofEnabled && dofCoCCoefficient > 0 && !camera.isOrthographic
            && dofs.dofShadingTilesPipeline != nil
    }

    /// The defocus shading rate's tile map: the least |CoC| of each 16×16 tile's geometry, under the
    /// DOF's own lens (the same `DOFParams` the gather will use this frame). Nil if it cannot run.
    private func encodeLightingRateTiles(_ cb: MTLCommandBuffer) -> MTLTexture? {
        guard let pipe = dofs.dofShadingTilesPipeline else { return nil }
        let tile = Self.dofTileSize
        let tileW = max(1, (width + tile - 1) / tile), tileH = max(1, (height + tile - 1) / tile)
        if dofs.lightingRateTileTexture?.width != tileW || dofs.lightingRateTileTexture?.height != tileH {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: tileW, height: tileH,
                                                             mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.lightingRateTileTexture = device.makeTexture(descriptor: d)
            dofs.lightingRateTileTexture?.label = "Illuminatorama.lighting.rateTiles"
        }
        guard let tiles = dofs.lightingRateTileTexture,
              let enc = timedComputeEncoder(cb, "lighting.rateTiles") else { return nil }
        enc.label = "Illuminatorama.lighting.rateTiles"
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        var p = DOFParams(
            invProjection: fu.invProjection,
            focusDist: max(0.05, dofFocusDistance),
            cocCoefficient: max(0, dofCoCCoefficient),
            maxRadius: max(1, dofMaxRadius),
            blades: dofBlades >= 3 ? Float(dofBlades) : 0,
            bladeRotation: dofBladeRotation,
            catsEye: max(0, min(1, dofCatsEye)),
            width: UInt32(width), height: UInt32(height),
            tileW: UInt32(tileW), tileH: UInt32(tileH), tileSize: UInt32(tile),
            prefilterScale: 0,
            cocFloor: max(0, dofDiffractionCoCPixels))
        enc.setComputePipelineState(pipe)
        enc.setTexture(gbs.depthTexture, index: 0)
        enc.setTexture(tiles, index: 1)
        enc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        let side = min(tile, max(1, Int(sqrt(Double(pipe.maxTotalThreadsPerThreadgroup)))))
        enc.dispatchThreadgroups(MTLSize(width: tileW, height: tileH, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: side, height: side, depth: 1))
        enc.endEncoding()
        return tiles
    }

    /// The coarse and skip lighting variants for this frame's feature bits (+ function constants 6 /
    /// 7). Compiled in the background on first use (nil meanwhile — the frame then lights every pixel,
    /// exactly as with the feature off); BLOCKING in a `blockingLightingCompile` lane, so a settled
    /// still is deterministic.
    private func lightingShadingRatePipelines() -> (coarse: MTLComputePipelineState, skip: MTLComputePipelineState)? {
        let rtSun = rtSunShadowLightingActive
        let aov = aovs.aovActiveThisFrame
        let bits: UInt8 = (iblEnabled ? 1 : 0) | (shadowsEnabled ? 2 : 0) | (dfgLUTEnabled ? 4 : 0)
                        | (rtSun ? 32 : 0)
                        | (aov ? 64 : 0)
        if bits == gbs.lastRatePipelines.flags, let memo = gbs.lastRatePipelines.pair { return memo }
        func variant(coarse: Bool) -> MTLComputePipelineState? {
            let (key, cv) = Self.lightingFeatureConstants(
                ibl: iblEnabled, shadow: shadowsEnabled, dfg: dfgLUTEnabled,
                rtSunShadow: rtSun, aov: aov)
            var c = coarse, k = !coarse
            cv.setConstantValue(&c, type: .bool, index: 6)
            cv.setConstantValue(&k, type: .bool, index: 7)
            let vkey = key + (coarse ? "-rateCoarse" : "-rateSkip")
            if blockingLightingCompile || rtSun || aov {
                return engine.pipelineCache.pipelineState(name: "illumi_lighting", device: device,
                                                          constants: cv, variantKey: vkey)
            }
            return engine.pipelineCache.pipelineStateAsync(name: "illumi_lighting", device: device,
                                                           constants: cv, variantKey: vkey)
        }
        guard let c = variant(coarse: true), let k = variant(coarse: false) else { return nil }
        gbs.lastRatePipelines = (bits, (c, k))
        return (c, k)
    }
}
