import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {

    /// Instanced-RT lighting pass — traces the TLAS. Mirrors `encodeRTLightingPass`
    /// but with per-instance albedo/normal lookups. No-op unless a TLAS is live.
    func encodeRTLightingTLASPass(_ cb: MTLCommandBuffer) {
        guard rts.rtTLASActive, let basePipeline = rts.rtTLASPipeline, let tlas = rts.rtTLAS,
              let instData = rts.rtInstanceDataBuffer, let objN = rts.rtObjNormalBuffer else { return }
        // Curve primitives (#60 item 7): when the TLAS contains curve
        // instances the intersector's geometry-type contract must include
        // curves, so the kRTCurvesEnabled variant is REQUIRED (rebuild only
        // appends curve instances when the variant compiled).
        let curvesOn = rts.rtCurveInstanceCount > 0
            && rts.rtCurveSetDataBuffer != nil && rts.rtCurvePoolPoints != nil
            && rts.rtCurvePoolRadii != nil && rts.rtCurvePoolSegments != nil
        let pipeline: MTLComputePipelineState
        if aovs.aovActiveThisFrame, let ap = rtTLASAOVVariant(curves: curvesOn) {
            pipeline = ap   // DH-1304 — the capture variant (the same kernel, its tags live)
        } else if curvesOn {
            guard let cp = rts.rtTLASCurvePipeline else { return }
            pipeline = cp
        } else {
            pipeline = basePipeline
        }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        rts.rtInstFrameSeed &+= 1
        var u = RTInstUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos,
            sunDir: simd_normalize(rtSunDirection), sunSoftnessRad: max(0.0005, rtSunSoftnessRad),
            sunColor: rtSunColor, giStrength: rtGIStrength,
            skyAmbient: ambientColor, specStrength: rtSpecStrength,
            width: UInt32(width), height: UInt32(height),
            shadowRays: UInt32(max(1, min(16, rtShadowRays))),
            giRays: UInt32(max(1, min(16, rtGIRays))),
            frameSeed: rts.rtInstFrameSeed, rayTMin: 0.004, maxGIDist: 60.0,
            reflStrength: max(0, rtReflStrength), reflMaxDist: max(0.1, rtReflMaxDistance),
            reflRoughnessCutoff: max(0, rtReflRoughnessCutoff),
            reflRays: UInt32(max(1, min(8, rtReflRays))),
            reflEnabled: rtReflectionsEnabled ? 1 : 0)
        // DH-0887 — the GI's progressive sample index: contiguous per dispatch of THIS pass.
        u.giProgressiveIndex = rts.rtGIProgressiveIndex
        u.surfaceSnapULPs = effectiveRTSurfaceSnapULPs
        u.surfaceSnapMinUpY = rtSurfaceSnapMinUpY          // DH-0952
        u.surfaceSnapClearance = rtSurfaceSnapClearance
        // DH-0952 — progressive Sobol cone (bit 0) and the still-only frame stride (bits 8–15); stride needs the raw add.
        let reflStride = (taaStillAccumulation && !rts.spec.enabled) ? UInt32(max(1, min(8, rtReflStillStride))) : 1
        let reflStrideLimit = UInt32(max(0, min(255, (rtReflStillStrideMaxRoughness * 255).rounded())))
        let reflStrideFLimit = UInt32(max(0, min(255, (rtReflStillStrideMaxFresnel * 255).rounded())))
        u.reflSampling = (rtReflProgressiveCone ? 1 : 0) | (reflStride << 8) | (reflStrideLimit << 16) | (reflStrideFLimit << 24)
        rts.rtGIProgressiveIndex &+= 1
        // DH-0896 — a reflection hit REPLACES the sky the deferred pass put there.
        let specIBLOn = rtReflectionsEnabled && ibls.specIBLWrittenThisFrame && ibls.specIBLTexture != nil
        u.reflReplacesIBL = specIBLOn ? 1 : 0
        let diffSkyOn = (rtGIReplacesDiffuseSky || rtPathTracedGI) && ibls.diffSkyWrittenThisFrame && ibls.diffSkyTexture != nil
        u.giReplacesDiffuseSky = diffSkyOn ? 1 : 0
        // DH-0989 — the path lane runs ONLY where the deferred pass handed its share over this
        // frame; otherwise its whole-indirect estimate would land on top of the stand-ins.
        let pathOn = rtPathTracedGI && diffSkyOn
        u.pathBounces = pathOn ? UInt32(max(1, min(32, rtPathBounces))) : 0
        u.pathClamp = pathOn ? max(0, rtPathClamp) : 0
        // DH-1011 — light sampling + MIS (bit 0), shadowed lamps (bit 1, needs bit 0), instrument (bit 7).
        u.pathFlags = !pathOn ? 0 : (rtPathLightSampling ? 1 : 0)
            | (rtPathLightSampling && rtPathLampShadows ? 2 : 0)
            | (rtPathLampBulbs ? 4 : 0)                       // DH-1295 — `kPathFlagLampBulbs`
            | (rtPathDebugFlags & 0x30F8)                       // + 4096: DH-1308's glossy instrument
        // DH-1014 — the layered weights apply to the one-bounce GI too, so NOT gated on `pathOn`.
        u.pathFlags |= (dfgLUTEnabled && layeredMultiscatter ? 256 : 0)
            | (dfgLUTEnabled && layeredClothSheen ? 512 : 0)
            | (dfgLUTEnabled && layeredClothSheen && layeredDyedSheen ? 262144 : 0)   // DH-1316 — `kPathFlagDyedSheen`
            | (diffuseRoughnessEnabled ? 16384 : 0)           // DH-1301 — `kPathFlagDiffuseRoughness`
            | (layeredSpecularEnabled ? 65536 : 0)            // DH-1300 — `kPathFlagLayeredSpecular`
            | (dfgLUTEnabled && layeredCoatEnabled ? 131072 : 0)   // DH-1309 — `kPathFlagLayeredCoat`
            | (thinFilmEnabled ? 524288 : 0)                  // DH-1014 — `kPathFlagThinFilm`
            | (!pathOn && rtGIHitFill ? 1024 : 0)            // DH-1320 — `kPathFlagOneBounceHitFill`
            // DH-1308 — `kPathFlagGlossyVertices`: only with light sampling + the DFG LUT (the lobe's albedo).
            | (pathOn && rtPathGlossyVertices && rtPathLightSampling && dfgLUTEnabled ? 2048 : 0)
            | (pathOn && rtPathThinTransmission ? 32768 : 0)   // Signack S5 — `kPathFlagThinTransmission`
        u.pathFlags |= rtTLASAblationFlags & 0x0F00_0000   // DH-0952 instrument (`kRTAblate*`)
        u.pathGlossyMinRoughness = pathOn ? max(0, rtPathGlossyMinRoughness) : 0
        u.pathGlossyLampSize = pathOn ? max(0, rtPathGlossyLampSize) : 1   // DH-1395
        u.layerInstanceCount = UInt32(instances.count)                       // DH-1300
        let specDenoiseOn = prepareSpecularDenoise()                         // DH-0539
        u.specDenoise = specDenoiseOn ? 1 : 0
        u.pathGlossyLampMinRoughness = pathOn ? max(0, rtPathGlossyLampMinRoughness) : 0
        // DH-1293 — the lamps' near-field levers (0 ⇒ the shipped point-source law).
        u.pathLampMinDist = pathOn ? rtPathLampMinDistance : 0
        u.pathLampSourceRadius = pathOn ? max(0, rtPathLampSourceRadius) : 0
        // DH-1344 — build (+cell) for the first frames after a reset, then use (−cell).
        if rts.rcNeedsClear { rts.rcFramesSinceReset = 0 }
        let rcCell = pathOn ? max(0, rtPathRadianceCacheCell) : 0
        u.radCacheCell = rcCell > 0 ? (rts.rcFramesSinceReset < rtPathRadianceCacheBuildFrames ? rcCell : -rcCell) : 0
        if rcCell > 0 { rts.rcFramesSinceReset += 1 }
        // A build frame needs one full walk per pixel, not `giRays` of them: the cache averages
        // millions of walks either way, and the reads are where the extra paths pay.
        if u.radCacheCell > 0 { u.giRays = min(u.giRays, UInt32(max(1, rtPathRadianceCacheBuildRays))) }
        // Surface cache read (P1c): on only when the grouped soup + cards + base
        // are all live this topology. The kernel gates every atlas read on this.
        let cacheOn = surfCacheActive && scs.surfCardCount > 0
            && rts.rtSoupTriBaseBuffer != nil && scs.surfTriCardBuffer != nil
            && surfConsumerAtlas != nil
            && rts.rtSoupTriBaseCount == instances.count   // base buffer matches this topology
        if cacheOn {
            u.surfCacheEnabled = 1
            u.surfTileSize = UInt32(scs.surfActiveTileSize)
            u.surfTilesPerRow = UInt32(scs.surfTilesPerRow)
            u.surfAtlasW = UInt32(scs.surfAtlasW); u.surfAtlasH = UInt32(scs.surfAtlasH)
            u.surfTriCount = UInt32(rts.rtTriangleCount)
        }
        // Debug isolation of the surface-cache term (see DebugTerm.surfaceCacheGI).
        u.debugSurfCacheGI = (debugTerm == .surfaceCacheGI) ? 1 : 0
        // Phase 5 / B0 — surface-cache variance heatmap (see DebugTerm.surfaceCacheVariance).
        u.debugSurfCacheVar = (debugTerm == .surfaceCacheVariance) ? 1 : 0
        // Phase 5 / A — residency feedback. The kernel marks `cardRequested[card]=1`
        // at each cache hit when enabled. Streaming (A1) NEEDS those marks, so it
        // implies feedback. The DRAIN (read + log + zero of the previous frame's
        // working set) is owned by exactly one site: `runStreamingResidency` (frame
        // start) when streaming is on, else the pure-feedback logger here. Draining in
        // both would zero the buffer before the residency pass could read it.
        let needFeedback = (surfaceCacheFeedback || surfaceCacheStreaming)
            && cacheOn && scs.surfCardRequestedBuffer != nil
        u.surfFeedbackEnabled = needFeedback ? 1 : 0
        if surfaceCacheFeedback, !surfaceCacheStreaming,
           let rb = scs.surfCardRequestedBuffer, scs.surfCardCount > 0 {
            awaitInFlightFrameBeforeCPUCacheWrite()   // the in-flight RT pass may still be marking cards
            let p = rb.contents().bindMemory(to: UInt32.self, capacity: scs.surfCardCount)
            var requested = 0
            for i in 0..<scs.surfCardCount where p[i] != 0 { requested += 1 }
            if requested > 0 {
                let pct = Int((Double(requested) / Double(scs.surfCardCount) * 100).rounded())
                Self.recordSurfCacheStats("feedback: requestedCards=\(requested) of \(scs.surfCardCount) (\(pct)% working set)")
            }
            memset(rb.contents(), 0, scs.surfCardCount * MemoryLayout<UInt32>.stride)
        }
        if curvesOn {
            u.curveInstanceBase = UInt32(instances.count)
            u.curveSetCount = UInt32(rts.rtCurveInstanceCount)
        }
        // ── Secondary-hit shading parity with the deferred pass ──────────────
        // A GI / reflection hit is now shaded by the SAME body the glass pass uses
        // (`shadeSecondarySurface`), so it gets the same inputs: the mesh UVs +
        // albedo atlas (a reflected plank floor used to be one flat colour), the
        // local point/spot lights, the cosine-convolved irradiance cube, and the
        // interior day-light separation. The three `glassHit*Enabled` switches gate
        // BOTH paths — they were never glass-specific, only glass-first.
        u.skyIntensity = max(0, iblIntensity)
        u.albedoAtlasEnabled = (glassHitTexturingEnabled
                                && rts.rtObjUVBuffer != nil && rts.rtObjUVCount > 0) ? 1 : 0
        u.objUVCount = UInt32(rts.rtObjUVCount)
        u.pointLightCount = glassHitLocalLightsEnabled ? UInt32(pointLights.count) : 0
        u.spotLightCount = glassHitLocalLightsEnabled ? UInt32(spotLights.count) : 0
        u.interiorMask = glassHitInteriorFillEnabled ? interiorLayerMask : 0
        u.interiorIBLUp = max(0, interiorIBLUp)
        u.interiorIBLSide = max(0, interiorIBLSide)
        u.interiorAmbient = max(0, interiorAmbient)
        // Interior irradiance bands — same values as the deferred pass, so a GI
        // bounce or reflection at an interior hit sees the same room it would on
        // screen (the lawn-green-ceiling fix, secondary side).
        let rtIrrW = max(0, min(1, interiorIrradianceWeight))
        u.interiorIrrUp = SIMD4(simd_max(interiorIrradianceUp, .zero), rtIrrW)
        u.setInteriorRoomGains(interiorRoomGains, enabled: interiorRoomGainsEnabled)
        // DH-0718 — area lights at secondary hits, in the room-gain meta's spare lanes.
        let rtAreaCount = secondaryAreaLightCount
        u.interiorRoomGainMeta.y = Float(rtAreaCount)
        u.interiorRoomGainMeta.z = Float(max(0, min(secondaryAreaShadowRays, 8)))
        u.interiorIrrSide = SIMD4(simd_max(interiorIrradianceSide, .zero), 0)
        u.interiorIrrDown = SIMD4(simd_max(interiorIrradianceDown, .zero), 0)
        // C1 — exactly one sun (see `RTSunOwnership`). This pass is ADDITIVE over a
        // complete deferred frame; with the old "compute direct whenever
        // shadowRays > 0" gate (default 4) and no deferred suppression, flipping
        // `rtEnabled` added a second sun. `.automatic` reproduces every existing
        // host; a host that keeps its deferred sun now gets GI + reflections only.
        u.directSunEnabled = rtOwnsDirectSun ? 1 : 0
        // C3 — shadow + GI rays see opaque (0x01) AND invisible occluders (0x04):
        // slabs that are real to light and never drawn. Camera-visible rays
        // (reflections here, refraction in the glass pass) deliberately don't.
        u.transportRayMask = Self.rtTransportRayMask
        // DH-0653 — chart-assignment overlay + hit/miss counters. The counters run with
        // the cache OFF too: their `reshade` column is the baseline a cache-on run's
        // `cached` column is compared against.
        u.debugSurfCacheCharts = (debugTerm == .surfaceCacheCharts) ? 1 : 0
        var hitStats: MTLBuffer? = nil
        if Self.surfCacheStatsRequested {
            scs.surfHitStatsBuffer = surfCounterBuffer(scs.surfHitStatsBuffer, slots: 8,
                                                   label: "Illuminatorama.surfcache.hitStats")
            if let sb = scs.surfHitStatsBuffer {
                if let line = Self.surfCacheHitLine(drainSurfCounters(sb, slots: 8)) {
                    Self.recordSurfCacheStats(line)
                }
                hitStats = sb
                u.surfStatsEnabled = 1
            }
        }
        // DH-0872 — same coefficient the tonemap pass uses, applied to the GI
        // pass's own raw sky-miss sample (see the kernel's miss branch).
        u.scotopicDesaturation = max(0, scotopicDesaturation)
        memcpy(rts.rtInstUniformBuffer.contents(), &u, MemoryLayout<RTInstUniforms>.stride)

        // DH-1344 — the radiance cache: allocated on first use, cleared with the GI history.
        let rcOn = u.radCacheCell != 0
        if rcOn {
            let cap = 1 << 21
            if rts.rcKeysBuffer == nil {
                rts.rcKeysBuffer = device.makeBuffer(length: cap * 4, options: .storageModePrivate)
                rts.rcDataBuffer = device.makeBuffer(length: cap * 16, options: .storageModePrivate)
                rts.rcNeedsClear = true
            }
            if rts.rcNeedsClear, let k = rts.rcKeysBuffer, let d = rts.rcDataBuffer, let blit = cb.makeBlitCommandEncoder() {
                blit.fill(buffer: k, range: 0..<k.length, value: 0)
                blit.fill(buffer: d, range: 0..<d.length, value: 0)
                blit.endEncoding()
                rts.rcNeedsClear = false
            }
        }
        if rts.rcDummyKeys == nil {
            rts.rcDummyKeys = device.makeBuffer(length: 16, options: .storageModePrivate)
            rts.rcDummyData = device.makeBuffer(length: 16, options: .storageModePrivate)
        }
        guard let enc = timedComputeEncoder(cb, "rtLightingTLAS") else { return }
        enc.setBuffer(rcOn ? rts.rcKeysBuffer : rts.rcDummyKeys, offset: 0, index: 21)
        enc.setBuffer(rcOn ? rts.rcDataBuffer : rts.rcDummyData, offset: 0, index: 22)
        enc.label = "Illuminatorama.rt.tlas"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(gbs.depthTexture, index: 0)
        enc.setTexture(gbs.gbufferNormalRgh, index: 1)
        enc.setTexture(gbs.gbufferAlbedoMet, index: 2)
        enc.setTexture(pfx.hdrCompositeTexture, index: 3)
        enc.setTexture(equirectSky ?? ibls.dummySkyTexture, index: 4)
        // Current radiance atlas (read at hits); dummy when the cache is off.
        enc.setTexture(cacheOn ? surfConsumerAtlas : (equirectSky ?? ibls.dummySkyTexture), index: 5)
        enc.setAccelerationStructure(tlas, bufferIndex: 0)
        enc.setBuffer(instData, offset: 0, index: 1)
        enc.setBuffer(objN, offset: 0, index: 2)
        enc.setBuffer(rts.rtInstUniformBuffer, offset: 0, index: 3)
        // Card buffers + per-instance soup base; dummies keep the binding valid
        // when the cache is off (the kernel never reads them then).
        enc.setBuffer(cacheOn ? scs.surfTriCardBuffer : instData, offset: 0, index: 4)
        enc.setBuffer(cacheOn ? scs.surfTriUVaBuffer : instData, offset: 0, index: 5)
        enc.setBuffer(cacheOn ? scs.surfTriUVcBuffer : instData, offset: 0, index: 6)
        enc.setBuffer(cacheOn ? rts.rtSoupTriBaseBuffer : instData, offset: 0, index: 7)
        enc.setBuffer(cacheOn ? scs.surfCardRectBuffer : instData, offset: 0, index: 8)  // per-card atlas rect
        enc.setBuffer(cacheOn ? scs.surfCardBuffer : (surfCardDummyBuffer ?? instData), offset: 0, index: 9)  // per-card material (L_out reconstruction); dummy must be ≥ 1 SurfCard for arg validation
        // Curve-set data + pooled geometry (#60 item 7); dummies keep the
        // bindings valid for the base (curve-free) pipeline variant.
        enc.setBuffer(curvesOn ? rts.rtCurveSetDataBuffer : instData, offset: 0, index: 10)
        enc.setBuffer(curvesOn ? rts.rtCurvePoolPoints : instData, offset: 0, index: 11)
        enc.setBuffer(curvesOn ? rts.rtCurvePoolRadii : instData, offset: 0, index: 12)
        enc.setBuffer(curvesOn ? rts.rtCurvePoolSegments : instData, offset: 0, index: 13)
        // Phase 5 / A — residency feedback (buffer 14). Bound always (kernel
        // signature requires it); written only when `surfFeedbackEnabled`. Dummy =
        // instData when the cache/feedback buffer isn't live (never written then).
        enc.setBuffer(scs.surfCardRequestedBuffer ?? instData, offset: 0, index: 14)
        // Secondary-hit shading parity (buffers 15–18, textures 6–7). Same
        // resources the glass pass binds for the same job; dummies keep the
        // bindings valid when a term is off (the uniforms gate every read).
        enc.setBuffer(rts.rtObjUVBuffer ?? instData, offset: 0, index: 15)
        enc.setBuffer(albedoAtlas.uvScaleBuffer, offset: 0, index: 16)
        enc.setBuffer(pointLightBuffer, offset: 0, index: 17)
        enc.setBuffer(spotLightBuffer, offset: 0, index: 18)
        enc.setBuffer(shd.spotProfileBuffer, offset: 0, index: 23)   // DH-0540 — the spot-profile rows
        enc.setBuffer(cloudShadowBinding, offset: 0, index: 25)      // DH-1462 — cloud sun transmittance
        // DH-0718 — area lights (buffer 20); a dummy when none are read.
        enc.setBuffer(secondaryAreaLightCount > 0 ? areaLightBuffer : instData, offset: 0, index: 20)
        enc.setTexture(ibls.irradianceCube, index: 6)
        enc.setTexture(albedoAtlas.texture, index: 7)
        // C2 — the noisy diffuse (soft shadow + GI) goes to its own buffer so the
        // temporal accumulator + SVGF/bilateral can clean it, exactly as the soup
        // path does. Before this the TLAS kernel composited it inline, which is
        // why both denoisers sat in the soup-only `else` branch doing nothing.
        enc.setTexture(rts.rtDiffuseTexture, index: 8)
        enc.setTexture(specIBLOn ? ibls.specIBLTexture : specIBLDummyTexture, index: 9)
        enc.setTexture(diffSkyOn ? ibls.diffSkyTexture : diffSkyDummyTexture, index: 10)
        enc.setTexture(ibls.dfgLUT, index: 11)              // DH-1014 — `layeredRTWeights`
        enc.setTexture(gbs.gbufferEmission, index: 12)     // DH-1014 — the cloth-sheen decode
        // DH-1300 — the material-layer slot target + the raster instance buffer it (and every TLAS id
        // below `layerInstanceCount`) indexes — IlluminatoramaMaterialLayers.h.
        enc.setTexture(gbs.gbufferMaterialLayer, index: 14)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 26)
        bindHitSunCascadeCull(enc)                         // DH-1588 — texture 17, buffers 27–28
        enc.setTexture(specDenoiseOn ? rts.spec.raw : rts.rtDiffuseTexture, index: 15)   // DH-0539 — the reflection's own target (a stand-in when off)
        enc.setTexture(specDenoiseOn ? rts.spec.fres : rts.rtDiffuseTexture, index: 16)
        // DH-0653 hit/miss counters (buffer 19). Dummy = instData, never added to while
        // `surfStatsEnabled` is 0.
        enc.setBuffer(hitStats ?? instData, offset: 0, index: 19)
        // DH-1304 — the AOV capture (texture 13 + buffer 24 exist only in the `aov` variants).
        if aovs.aovActiveThisFrame, let aovTex = aovs.aovTexture {
            enc.setTexture(aovTex, index: 13)
            var au = AOVUniforms(blend: aovs.aovFrameBlend, handoff: diffSkyOn ? 1 : 0)
            enc.setBytes(&au, length: MemoryLayout<AOVUniforms>.stride, index: 24)
        }
        // The TLAS references the BLASes, which reference the mesh vertex/index
        // buffers (and the curve BLASes the pooled curve buffers) — all must be
        // resident for the intersector.
        for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
        for blas in rts.rtCurveBLASList { enc.useResource(blas, usage: .read) }
        for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
        dispatch(enc, pipeline: pipeline, width: width, height: height,
                 maxThreadsOverride: rtThreadgroupMax)
        enc.endEncoding()
    }

    /// Hardware ray-traced sun lighting (soft shadows + 1-bounce GI) added into
    /// the HDR composite. Runs after SSR so the additive result feeds TAA /
    /// bloom / tonemap on the same path as everything else. No-op unless
    /// `rtEnabled`, the device supports RT, and a geometry/AS has been built.
    func encodeRTLightingPass(_ cb: MTLCommandBuffer) {
        guard rtEnabled, rtSupported,
              let basePipeline = rts.rtPipeline, let accel = rts.rtAccel,
              let triA = rts.rtTriAlbedoBuffer, let triN = rts.rtTriNormalBuffer,
              let vb = rts.rtVertexBuffer, let ib = rts.rtIndexBuffer,
              rts.rtTriangleCount > 0 else { return }
        // Curve primitives (#60 item 7 incr. 2): when the soup AS holds a curve
        // geometry descriptor, the intersector's geometry-type contract must
        // include curves, so the kRTCurvesEnabled variant is REQUIRED.
        let curvesOn = rts.rtSoupCurvesActive && rts.rtSoupCurvePipeline != nil
            && rts.rtSoupCurveSetDataBuffer != nil && rts.rtSoupCurvePoolPoints != nil
            && rts.rtSoupCurveRadii != nil && rts.rtSoupCurveSegments != nil
        let pipeline = curvesOn ? rts.rtSoupCurvePipeline! : basePipeline

        // Pull the (jittered) invVP + camera position the G-buffer used so the
        // world reconstruction matches the deferred passes exactly.
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        rts.rtFrameSeed &+= 1
        var u = RTUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos,
            sunDir: simd_normalize(rtSunDirection), sunSoftnessRad: max(0.0005, rtSunSoftnessRad),
            sunColor: rtSunColor, giStrength: rtGIStrength,
            skyAmbient: ambientColor, specStrength: rtSpecStrength,
            width: UInt32(width), height: UInt32(height),
            shadowRays: UInt32(max(1, min(16, rtShadowRays))),
            giRays: UInt32(max(1, min(16, rtGIRays))),
            frameSeed: rts.rtFrameSeed, rayTMin: 0.004, maxGIDist: 60.0,
            triangleCount: UInt32(rts.rtTriangleCount),
            reflStrength: max(0, rtReflStrength),
            reflMaxDist: max(0.1, rtReflMaxDistance),
            reflRoughnessCutoff: max(0, rtReflRoughnessCutoff),
            reflRays: UInt32(max(1, min(8, rtReflRays))),
            reflEnabled: rtReflectionsEnabled ? 1 : 0,
            surfCacheEnabled: surfCacheActive ? 1 : 0,
            // Must be the ACTIVE tile size the atlas + update pass use (6 for
            // per-triangle cards, 64 for the room), NOT the 64 default — else the
            // read samples at the wrong scale and bleeds across tiles.
            surfTileSize: UInt32(scs.surfActiveTileSize), surfTilesPerRow: UInt32(scs.surfTilesPerRow),
            surfAtlasW: UInt32(scs.surfAtlasW), surfAtlasH: UInt32(scs.surfAtlasH),
            leafTransmission: max(0, leafTransmission))
        u.curvesEnabled = curvesOn ? 1 : 0
        // C1 — exactly one sun. See `RTSunOwnership`; `.automatic` (the default)
        // reproduces every existing soup host, all of which zero
        // `directionalLightColor` before enabling RT.
        u.directSunEnabled = rtOwnsDirectSun ? 1 : 0
        // DH-0872 — same coefficient the tonemap pass uses, applied to the GI
        // pass's own raw sky-miss sample (see the kernel's miss branch).
        u.scotopicDesaturation = max(0, scotopicDesaturation)
        memcpy(rts.rtUniformBuffer.contents(), &u, MemoryLayout<RTUniforms>.stride)

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.rt"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(gbs.depthTexture, index: 0)
        enc.setTexture(gbs.gbufferNormalRgh, index: 1)
        enc.setTexture(gbs.gbufferAlbedoMet, index: 2)
        enc.setTexture(pfx.hdrCompositeTexture, index: 3)
        enc.setTexture(equirectSky ?? ibls.dummySkyTexture, index: 4)
        // Surface-cache atlas (current) + per-triangle card-UV maps. Always
        // bound; the kernel gates reads on `surfCacheEnabled`. Dummies keep
        // Metal happy when the cache is off / unbuilt.
        enc.setTexture(surfConsumerAtlas ?? ibls.dummySkyTexture, index: 5)
        enc.setTexture(rts.rtDiffuseTexture, index: 6)   // diffuse (shadow+GI) → denoise pass
        enc.setAccelerationStructure(accel, bufferIndex: 0)
        enc.setBuffer(triA, offset: 0, index: 1)
        enc.setBuffer(triN, offset: 0, index: 2)
        enc.setBuffer(rts.rtUniformBuffer, offset: 0, index: 3)
        enc.setBuffer(scs.surfTriCardBuffer ?? ib, offset: 0, index: 4)
        enc.setBuffer(scs.surfTriUVaBuffer ?? triA, offset: 0, index: 5)
        enc.setBuffer(scs.surfTriUVcBuffer ?? triA, offset: 0, index: 6)
        enc.setBuffer(scs.surfCardRectBuffer ?? triA, offset: 0, index: 8) // per-card atlas rect (dummy off)
        enc.setBuffer(scs.surfCardBuffer ?? surfCardDummyBuffer ?? triA, offset: 0, index: 9)     // per-card material (L_out reconstruction; dummy off, ≥ 1 SurfCard for arg validation)
        // Curve set data + pooled geometry (#60 item 7 incr. 2); dummies keep the
        // bindings valid for the base (curve-free) pipeline variant.
        enc.setBuffer(curvesOn ? rts.rtSoupCurveSetDataBuffer : triA, offset: 0, index: 10)
        enc.setBuffer(curvesOn ? rts.rtSoupCurvePoolPoints : triA, offset: 0, index: 11)
        enc.setBuffer(curvesOn ? rts.rtSoupCurveRadii : triA, offset: 0, index: 12)
        enc.setBuffer(curvesOn ? rts.rtSoupCurveSegments : triA, offset: 0, index: 13)
        enc.setBuffer(cloudShadowBinding, offset: 0, index: 15)      // DH-1492 — cloud sun transmittance
        // Curve-only primitive AS (#60 item 7 incr. 2) — a SECOND acceleration
        // structure the kernel traces alongside the triangle soup. Bind the
        // triangle `accel` as a harmless dummy for the base (curve-free) variant.
        enc.setAccelerationStructure((curvesOn ? rts.rtSoupCurveAccel : nil) ?? accel, bufferIndex: 14)
        // The intersector reads triangle vertex positions (and, when curves are
        // active, the curve control points) through the AS; Metal needs explicit
        // residency hints for the buffers it can't statically prove are referenced.
        enc.useResource(vb, usage: .read)
        enc.useResource(ib, usage: .read)
        if curvesOn {
            if let ca = rts.rtSoupCurveAccel { enc.useResource(ca, usage: .read) }
            if let p = rts.rtSoupCurvePoolPoints { enc.useResource(p, usage: .read) }
            if let r = rts.rtSoupCurveRadii { enc.useResource(r, usage: .read) }
            if let s = rts.rtSoupCurveSegments { enc.useResource(s, usage: .read) }
        }
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// **True when an RT lighting pass wrote `rtDiffuseTexture` this frame** — the one
    /// precondition every consumer of that buffer shares, in one place so they cannot
    /// drift apart (C2: they used to guard on the SOUP path's resources, which is why
    /// nothing denoised the TLAS path — `rtPipeline`/`rtAccel` are nil there).
    private var rtDiffuseWrittenThisFrame: Bool {
        guard rtEnabled, rtSupported else { return false }
        // A live TLAS routes to `encodeRTLightingTLASPass`, which only runs (and only
        // writes the buffer) when the opaque lighting pass is enabled. A glass-only
        // TLAS therefore correctly reports false.
        if rts.rtTLASActive { return rtOpaqueLightingEnabled }
        return rts.rtPipeline != nil && rts.rtAccel != nil && rts.rtTriangleCount > 0
    }

    /// Clean the RT diffuse (soft shadow + 1-bounce GI) and composite it. Shared by
    /// BOTH lighting paths — see the frame-graph call sites. Order matters: temporal
    /// accumulation first (it feeds the spatial filter a far less noisy input), then
    /// either the SVGF à-trous cascade or the fixed-radius bilateral.
    func encodeRTDiffuseDenoiseChain(_ cb: MTLCommandBuffer) {
        // Temporally accumulate the RT diffuse (GI) so it keeps converging under
        // camera motion (kills the wall crawl) BEFORE the spatial denoise reads it.
        // No-op unless rtGITemporalEnabled.
        encodeRTGITemporalAccum(cb)
        // Phase 4.44 — SVGF à-trous cascade replaces the fixed-radius bilateral when
        // enabled. Falls back to the original bilateral path.
        if effectiveSVGFEnabled && rtGITemporalEnabled && rtEnabled && rtSupported {
            encodeSVGFCascade(cb)
        } else {
            encodeRTDenoiseComposite(cb)
        }
    }

    /// Temporal accumulation of the RT diffuse (1-bounce GI + soft shadow)
    /// term: velocity-reprojected exponential history so the GI keeps
    /// converging under camera motion. Reads the raw RT diffuse + previous
    /// accumulated history, writes the new accumulated history (which the
    /// denoise then reads via `rtDiffuseDenoiseSource`, and which becomes next
    /// frame's history after the per-frame toggle). No-op unless enabled.
    private func encodeRTGITemporalAccum(_ cb: MTLCommandBuffer) {
        guard rtGITemporalEnabled, rtDiffuseWrittenThisFrame else { return }
        var u = RTGITemporalUniforms(
            width: UInt32(width), height: UInt32(height),
            enabled: 1,
            isFirstFrame: rtGINeedsFirstFrame ? 1 : 0,
            blend: max(0.01, min(1.0, rtGITemporalBlend)),
            gammaClamp: max(1.0, rtGITemporalClamp),
            staticAccumulation: (rtGIStaticAccumulation ? 1 : 0) | (giIrradianceMode ? 2 : 0)
                | (rtGIStaticAccumulation && rtGIStillPerFrame && taaStillAccumulation ? 4 : 0))
        guard let enc = timedComputeEncoder(cb, "rt.giTemporal") else { return }
        enc.label = "Illuminatorama.rt.giTemporal"
        enc.setComputePipelineState(rts.rtGITemporalPipeline)
        enc.setTexture(rts.rtDiffuseTexture,           index: 0)  // current raw GI
        enc.setTexture(previousRTGIHistoryTexture, index: 1)  // history read
        enc.setTexture(pfx.velocityTexture,            index: 2)
        enc.setTexture(currentRTGIHistoryTexture,  index: 3)  // accumulated write
        enc.setTexture(rts.giSampleCount,              index: 4)  // adaptive count (read_write)
        enc.setTexture(gbs.gbufferAlbedoMet,           index: 5)  // still: irradiance demodulation (DH-0966)
        // `setBytes`, not the shared single buffer: with two frames in flight a memcpy into one
        // buffer can hand the GPU the NEXT frame's `isFirstFrame` / mode (DH-0887 audit).
        enc.setBytes(&u, length: MemoryLayout<RTGITemporalUniforms>.stride, index: 0)
        dispatch(enc, pipeline: rts.rtGITemporalPipeline, width: width, height: height)
        enc.endEncoding()
    }

    // ── Phase 4.44: SVGF à-trous cascade ─────────────────────────────────────

    private struct SVGFAtrousUniforms {
        /// `flags` (DH-0966): bit 0 ⇒ the input is irradiance; multiply the centre albedo back in (last level).
        var width: UInt32; var height: UInt32; var stepSize: UInt32; var flags: UInt32 = 0
        var sigmaL: Float; var sigmaZ: Float; var sigmaN: Float; var lumFloor: Float = 0
        /// DH-1016 — the variance gate (0 = off); see `svgfStillVarianceGate`.
        var varGate: Float = 0
    }

    /// SVGF denoiser for the RT GI term: variance estimate + N à-trous cascade
    /// levels. Reads `currentRTGIHistoryTexture` (the temporally-accumulated GI)
    /// and writes the final spatially-filtered result into `giAtrousA/B`, then
    /// adds it additively into `hdrCompositeTexture`.
    ///
    /// Only runs when `svgfEnabled && rtGITemporalEnabled && rtEnabled`.
    /// Replaces `encodeRTDenoiseComposite` for the GI term on this path.
    private func encodeSVGFCascade(_ cb: MTLCommandBuffer) {
        guard effectiveSVGFEnabled, rtGITemporalEnabled, rtDiffuseWrittenThisFrame else { return }

        let levels = max(1, min(5, effectiveSVGFLevels))
        let W = UInt32(width), H = UInt32(height)

        // Pass 1: variance estimate from the accumulated GI.
        do {
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "Illuminatorama.svgf.variance"
            enc.setComputePipelineState(rts.svgfVariancePipeline)
            enc.setTexture(currentRTGIHistoryTexture, index: 0)  // accumulated GI
            enc.setTexture(rts.giVariance,                index: 1)  // variance output
            var u = SVGFAtrousUniforms(width: W, height: H, stepSize: 1,
                                        sigmaL: svgfSigmaL, sigmaZ: svgfSigmaZ, sigmaN: svgfSigmaN)
            enc.setBytes(&u, length: MemoryLayout<SVGFAtrousUniforms>.stride, index: 0)
            dispatch(enc, pipeline: rts.svgfVariancePipeline, width: width, height: height)
            enc.endEncoding()
        }

        // À-trous cascade. Colour ping-pongs {giAtrousA, giAtrousB}; variance
        // ping-pongs {giVariance, giVarianceB} — the two pairs are DISJOINT so no
        // pass ever reads one signal at dilated neighbour taps while writing the
        // other to the same texture at self. The old layout reused giAtrousB for
        // both colour and variance, so each level≥1 had an in-place read/write race
        // that corrupted colour in a dilated-stride pattern → the masonry (#65).
        //
        // Trace:
        //   level 0: cIn=giHistory  vIn=giVariance  → cOut=giAtrousA  vOut=giVarianceB
        //   level 1: cIn=giAtrousA  vIn=giVarianceB → cOut=giAtrousB  vOut=giVariance
        //   level 2: cIn=giAtrousB  vIn=giVariance  → cOut=giAtrousA  vOut=giVarianceB
        //   … colour and variance never share a texture. Result in `lastColorOut`.
        var lastColorOut = rts.giAtrousA!
        for level in 0..<levels {
            let stepSize = UInt32(1 << level)
            let cIn  = level == 0 ? currentRTGIHistoryTexture : (level % 2 == 1 ? rts.giAtrousA : rts.giAtrousB)
            let cOut = level % 2 == 0 ? rts.giAtrousA  : rts.giAtrousB
            let vIn  = level % 2 == 0 ? rts.giVariance : rts.giVarianceB
            let vOut = level % 2 == 0 ? rts.giVarianceB : rts.giVariance
            lastColorOut = cOut!

            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "Illuminatorama.svgf.atrous.\(level)"
            enc.setComputePipelineState(rts.svgfAtrousPipeline)
            enc.setTexture(cIn,              index: 0)
            enc.setTexture(vIn,              index: 1)
            enc.setTexture(gbs.depthTexture,     index: 2)
            enc.setTexture(gbs.gbufferNormalRgh, index: 3)
            enc.setTexture(cOut,             index: 4)
            enc.setTexture(vOut,             index: 5)
            enc.setTexture(gbs.gbufferAlbedoMet, index: 6)
            let flags: UInt32 = (giIrradianceMode && level == levels - 1 ? 1 : 0)
                | (!svgfStillLuminanceStop && rtGIStaticAccumulation ? 4 : 0)
            var u = SVGFAtrousUniforms(width: W, height: H, stepSize: stepSize, flags: flags,
                                        sigmaL: svgfSigmaL, sigmaZ: svgfSigmaZ, sigmaN: svgfSigmaN,
                                        lumFloor: effectiveSVGFLumFloor,
                                        varGate: rtGIStaticAccumulation ? svgfStillVarianceGate : 0)
            enc.setBytes(&u, length: MemoryLayout<SVGFAtrousUniforms>.stride, index: 0)
            enc.setBuffer(frameUniformBuffer, offset: 0, index: 1)  // invProjection for linear depth
            dispatch(enc, pipeline: rts.svgfAtrousPipeline, width: width, height: height)
            enc.endEncoding()
        }

        addGIIntoComposite(cb, source: lastColorOut)
    }

    /// Additively composites the denoised GI buffer into `hdrCompositeTexture`.
    /// Reuses `illumi_rt_denoise` with radius=0 (no bilateral — SVGF already filtered).
    private func addGIIntoComposite(_ cb: MTLCommandBuffer, source: MTLTexture) {
        var du = RTDenoiseUniforms(width: UInt32(width), height: UInt32(height),
                                   enabled: 0, radius: 0,
                                   kDepth: 0, kNorm: 0)
        memcpy(rts.rtDenoiseUniformBuffer.contents(), &du, MemoryLayout<RTDenoiseUniforms>.stride)
        guard let enc = timedComputeEncoder(cb, "svgf.composite") else { return }
        enc.label = "Illuminatorama.svgf.composite"
        enc.setComputePipelineState(rts.rtDenoisePipeline)
        enc.setTexture(source,              index: 0)
        enc.setTexture(gbs.depthTexture,        index: 1)
        enc.setTexture(gbs.gbufferNormalRgh,    index: 2)
        enc.setTexture(pfx.hdrCompositeTexture, index: 3)
        enc.setBuffer(frameUniformBuffer,       offset: 0, index: 0)
        enc.setBuffer(rts.rtDenoiseUniformBuffer,   offset: 0, index: 1)
        dispatch(enc, pipeline: rts.rtDenoisePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// Depth + normal guided bilateral over the RT diffuse buffer, adding the
    /// cleaned soft-shadow + 1-bounce-GI term into the HDR composite (additive,
    /// own-pixel read-modify-write) before the volumetric / TAA passes. Gated on
    /// `rtDiffuseWrittenThisFrame`, so it only composites a buffer some lighting
    /// pass actually wrote — EITHER path, since C2; with `rtDenoiseEnabled` off it
    /// passes the raw diffuse straight through (the old additive behaviour).
    private func encodeRTDenoiseComposite(_ cb: MTLCommandBuffer) {
        guard rtDiffuseWrittenThisFrame else { return }
        var u = RTDenoiseUniforms(
            width: UInt32(width), height: UInt32(height),
            enabled: rtDenoiseEnabled ? 1 : 0,
            radius: UInt32(max(0, min(4, rtDenoiseRadius))),
            kDepth: max(0, rtDenoiseDepthSensitivity),
            kNorm: max(0, rtDenoiseNormalSharpness))
        memcpy(rts.rtDenoiseUniformBuffer.contents(), &u, MemoryLayout<RTDenoiseUniforms>.stride)
        guard let enc = timedComputeEncoder(cb, "rt.denoise") else { return }
        enc.label = "Illuminatorama.rt.denoise"
        enc.setComputePipelineState(rts.rtDenoisePipeline)
        enc.setTexture(rtDiffuseDenoiseSource, index: 0)
        enc.setTexture(gbs.depthTexture,        index: 1)
        enc.setTexture(gbs.gbufferNormalRgh,    index: 2)
        enc.setTexture(pfx.hdrCompositeTexture, index: 3)
        enc.setBuffer(frameUniformBuffer,     offset: 0, index: 0)
        enc.setBuffer(rts.rtDenoiseUniformBuffer, offset: 0, index: 1)
        dispatch(enc, pipeline: rts.rtDenoisePipeline, width: width, height: height)
        enc.endEncoding()
    }
}

// MARK: - DH-1588 — the sun cascades as a shadow proof at GI / reflection hits

/// Mirror of `RTSunCascadeCull` (IlluminatoramaRTInstanced.metal).
struct RTSunCascadeCull {
    var vp0: simd_float4x4, vp1: simd_float4x4, vp2: simd_float4x4
    var enabled: UInt32, mapSize: UInt32, marginMetres: Float, slopeTexels: Float
    var audit: UInt32, minNdotL: Float, pad0: UInt32 = 0, pad1: UInt32 = 0
}

extension IlluminatoramaRenderer {

    /// What the audit instrument counted since the last reset (`rtHitSunCascadeAudit`): GI / reflection
    /// hits facing the sun that were asked, how many the cascades proved shadowed, and how many of those
    /// the traced ray then found UNoccluded (the proof's false shadows). Read after a blocking render.
    public var rtHitSunCascadeCullStats: (asked: UInt32, proven: UInt32, falseShadow: UInt32) {
        guard let b = rts.hitSunCullStats else { return (0, 0, 0) }
        let p = b.contents().bindMemory(to: UInt32.self, capacity: 4)
        return (p[0], p[1], p[2])
    }

    public func resetRTHitSunCascadeCullStats() {
        guard let b = rts.hitSunCullStats else { return }
        memset(b.contents(), 0, 16)
    }

    /// Texture 17 (the cascade depth array — always bound; a cleared slice reads 1.0 and proves nothing),
    /// buffer 27 (the `RTSunCascadeCull` block) and buffer 28 (the audit counters).
    func bindHitSunCascadeCull(_ enc: MTLComputeCommandEncoder) {
        if rts.hitSunCullStats == nil {
            rts.hitSunCullStats = device.makeBuffer(length: 16, options: .storageModeShared)
            rts.hitSunCullStats?.label = "Illuminatorama.rt.hitSunCullStats"
            resetRTHitSunCascadeCullStats()
        }
        let on = rtHitSunCascadeCull && shadowsEnabled && shd.shadowMap != nil && shd.cascadeVPs.count >= 3
        var cc = RTSunCascadeCull(
            vp0: on ? shd.cascadeVPs[0] : matrix_identity_float4x4,
            vp1: on ? shd.cascadeVPs[1] : matrix_identity_float4x4,
            vp2: on ? shd.cascadeVPs[2] : matrix_identity_float4x4,
            enabled: on ? 1 : 0, mapSize: UInt32(shd.shadowMap?.width ?? 1),
            marginMetres: max(0, rtHitSunCascadeMargin), slopeTexels: max(0, rtHitSunCascadeSlopeTexels),
            audit: on && rtHitSunCascadeAudit ? 1 : 0, minNdotL: 0.1)
        enc.setTexture(shd.shadowMap, index: 17)
        enc.setBytes(&cc, length: MemoryLayout<RTSunCascadeCull>.stride, index: 27)
        enc.setBuffer(rts.hitSunCullStats, offset: 0, index: 28)
    }
}
