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
            | (!pathOn && rtGIHitFill ? 1024 : 0)            // DH-1320 — `kPathFlagOneBounceHitFill`
            // DH-1308 — `kPathFlagGlossyVertices`: only with light sampling + the DFG LUT (the lobe's albedo).
            | (pathOn && rtPathGlossyVertices && rtPathLightSampling && dfgLUTEnabled ? 2048 : 0)
        u.pathGlossyMinRoughness = pathOn ? max(0, rtPathGlossyMinRoughness) : 0
        u.pathGlossyLampSize = pathOn ? max(0, rtPathGlossyLampSize) : 1   // DH-1395
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
                                        lumFloor: effectiveSVGFLumFloor)
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

// MARK: - RTState

/// Hardware ray tracing: acceleration structures, TLAS/curve pools, GI history + denoise, radiance cache, RT uniform buffers.
/// Owned by the renderer as `rts`; every member keeps the name it had on the renderer.
@MainActor
final class RTState {
    /// The unjittered view-projection of the last frozen-camera resolve (see the camera-cut guard
    /// in `render`); nil while still mode is off.
    var lastStillViewProjection: simd_float4x4?
    var previousRTGIStatic = false

    var rcFramesSinceReset = 0
    var rcKeysBuffer: MTLBuffer?
    var rcDataBuffer: MTLBuffer?
    var rcDummyKeys: MTLBuffer?
    var rcDummyData: MTLBuffer?

    // RT diffuse (soft shadow + 1-bounce GI) buffer — written by the RT pass,
    // bilateral-filtered into the composite by `encodeRTDenoiseComposite`.
    var rtDiffuseTexture: MTLTexture!

    // RT-GI temporal accumulation: the rtDiffuse term gets its own
    // velocity-reprojected exponential history so the 1-bounce GI keeps
    // converging under camera motion (the main TAA can't). Ping-pong, full-res.
    var rtGIHistoryA: MTLTexture!  // full-res, rgba16Float
    var rtGIHistoryB: MTLTexture!
    var rtGIHistoryToggle: Bool = false

    var rcNeedsClear = true
    var previousRTGITemporalEnabled: Bool = false

    var giSampleCount: MTLTexture!  // full-res
    // SVGF à-trous cascade textures for RT GI (when svgfEnabled).
    // Color ping-pongs {giAtrousA, giAtrousB}; variance ping-pongs {giVariance,
    // giVarianceB}. The two pairs MUST stay disjoint — an à-trous level reads one
    // signal at dilated neighbour taps while writing the other to its own texel, so
    // any shared texture is an in-place read/write race that corrupts colour in a
    // dilated-stride pattern (the issue-#65 masonry). Three buffers can't keep both
    // ping-pongs disjoint; hence the fourth.
    var giVariance: MTLTexture!  // full-res r16Float (variance A)
    var giVarianceB: MTLTexture!  // full-res r16Float (variance B)
    var giAtrousA: MTLTexture!  // full-res rgba16Float
    var giAtrousB: MTLTexture!  // full-res rgba16Float

    var previousGIIrradianceMode = false

    var rtDenoisePipeline: MTLComputePipelineState!
    var rtGITemporalPipeline: MTLComputePipelineState!
    var svgfVariancePipeline: MTLComputePipelineState!  // Phase 4.44 SVGF
    var svgfAtrousPipeline: MTLComputePipelineState!  // Phase 4.44 SVGF

    var rtProxyMeshKinds: Set<IlluminatoramaRenderer.MeshKind> = []

    var rtPipeline: MTLComputePipelineState?
    var rtAccel: MTLAccelerationStructure?
    var rtScratch: MTLBuffer?
    var rtVertexBuffer: MTLBuffer?
    var rtIndexBuffer: MTLBuffer?
    var rtTriAlbedoBuffer: MTLBuffer?
    var rtTriNormalBuffer: MTLBuffer?
    var rtTriangleCount: Int = 0
    var rtUniformBuffer: MTLBuffer!
    var rtDenoiseUniformBuffer: MTLBuffer!
    var rtGITemporalUniformBuffer: MTLBuffer!
    var rtFrameSeed: UInt32 = 0

    var rtTLASPipeline: MTLComputePipelineState?

    var rtMeshCache: [ObjectIdentifier: IlluminatoramaRenderer.RTMeshEntry] = [:]
    /// Whether any cached mesh needs the per-frame refit. Stored rather than derived so the
    /// per-frame static-skip test stays O(1); maintained by `pruneRTMeshCache()`, which runs
    /// on every rebuild — the only occasion the answer can change.
    var rtHasDeformingBLAS = false

    var rtResidentBuffers: [MTLBuffer] = []   // mesh vb/ib the BLASes reference
    var rtBLASList: [MTLAccelerationStructure] = []   // index = accelerationStructureIndex
    var rtObjNormalBuffer: MTLBuffer?  // concatenated object-space per-tri normals (float4)
    var rtObjNormalCount: Int = 0
    /// Concatenated per-triangle MESH UVs (3 × `float2` per triangle), built in
    /// lockstep with `rtObjNormalBuffer` so a hit reads its texture coordinates at
    /// `(normalBase + primitive_id) * 3`. Lets a ray-traced hit take a REAL albedo
    /// atlas sample instead of the instance's mean albedo — the through-glass
    /// "flat slabs" fix. Nil when no mesh exposed CPU-readable vertices.
    var rtObjUVBuffer: MTLBuffer?
    var rtObjUVCount: Int = 0

    // Surface cache (P1c): per-TLAS-instance base offset into the grouped soup
    // triangle list (== the `triCard`/`triUVa`/`triUVc` index space). Built
    // alongside the world-space soup in `rebuildRTAccel` when the cache is on,
    // so a TLAS hit resolves `soupTriBase[instance_id] + primitive_id` → card.
    var rtSoupTriBaseBuffer: MTLBuffer?
    var rtSoupTriBaseCount: Int = 0

    var rtInstanceDataBuffer: MTLBuffer?  // RTInstanceData[], grouped order
    var rtInstanceDescBuffer: MTLBuffer?  // MTLAccelerationStructureInstanceDescriptor[]
    var rtTLAS: MTLAccelerationStructure?
    var rtTLASScratch: MTLBuffer?
    var rtTLASCapacity: Int = 0
    var rtTLASInstanceCount: Int = 0
    var rtTLASTopologyHash: Int = .min
    // ── Curve primitives (#60 item 7, increment 1) ──
    // Sets adopted from `IlluminatoramaCurveRegistry` (synced per frame by
    // version). Control points / radii / segment indices for ALL sets are
    // pooled into three shared buffers (segment index values are ABSOLUTE into
    // the pool, so every curve BLAS and the shading kernels read the same
    // pool with only a per-set segment base). One curve BLAS per set; one TLAS
    // instance per set, appended AFTER the triangle instances — a curve hit's
    // `instance_id - curveInstanceBase` is its set index. Kernels that trace
    // the TLAS switch to a `kRTCurvesEnabled` function-constant variant when
    // curves are present (the intersector's geometry-type contract must match
    // the AS content), so curve-free scenes keep the exact pipelines they had.
    var rtCurveSets: [IlluminatoramaCurveSet] = []
    var rtCurveSyncVersion: Int = 0
    var rtCurveBLASList: [MTLAccelerationStructure] = []
    var rtCurvePoolPoints: MTLBuffer?  // packed float3 (stride 12)
    var rtCurvePoolRadii: MTLBuffer?  // float
    var rtCurvePoolSegments: MTLBuffer?  // uint32, absolute into the pool
    var rtCurvePoolPointCount: Int = 0
    var rtCurveSetSegmentBase: [Int] = [] // per set: first segment in the pool
    var rtCurveSetDataBuffer: MTLBuffer?  // RTCurveSetData per set
    var rtCurveInstanceCount: Int = 0     // sets in the CURRENT TLAS topology
    var rtTLASCurvePipeline: MTLComputePipelineState?

    // ── Curve primitives on the SOUP path (#60 item 7, increment 2) ──
    // Forest et al. are native soup scenes (`setRTGeometry` → primitive AS,
    // `encodeRTLightingPass`), not the TLAS path. The soup AS gains a SECOND
    // geometry descriptor (a curve geometry) alongside the triangle soup; a hit
    // is a curve iff `res.type == curve` (single set, identity-placed, so no
    // instance_id). The pooled rest points are re-displaced per frame by the
    // wind kernel and the AS is refit, so the RT wood sways with the raster tubes.
    var rtSoupCurveSet: IlluminatoramaCurveSet?
    var rtSoupCurveSyncVersion: Int = 0
    var rtSoupCurvesActive: Bool = false
    var rtSoupCurveRestPoints: MTLBuffer?  // packed float3, un-swayed (kernel input)
    var rtSoupCurvePoolPoints: MTLBuffer?  // packed float3, AS-referenced (kernel output / rest copy)
    var rtSoupCurveRadii: MTLBuffer?  // float
    var rtSoupCurveSegments: MTLBuffer?  // uint32 (relative — single geometry descriptor)
    var rtSoupCurveWindAttr: MTLBuffer?  // float4 per point (empty set ⇒ nil ⇒ rigid)
    var rtSoupCurveSetDataBuffer: MTLBuffer?  // single RTCurveSetData
    var rtSoupCurvePointCount: Int = 0
    var rtSoupCurveSegmentCount: Int = 0
    /// SEPARATE curve-only primitive AS (#60 item 7 incr. 2). Metal forbids mixing
    /// triangle + curve geometry descriptors in ONE primitive AS ("must be the
    /// same type"), so curves live in their own AS and the kernel traces BOTH
    /// (any-hit occludes; nearest-hit compares distance). `.refit` usage for sway.
    var rtSoupCurveAccel: MTLAccelerationStructure?
    var rtSoupRefitScratch: MTLBuffer?
    var rtSoupCurvePipeline: MTLComputePipelineState?

    var curveWindDisplacePipeline: MTLComputePipelineState?

    /// Freeze curves at rest pose (A/B for the sway refit). `VIZ_ILLUMI_NO_CURVE_REFIT=1`.
    let noCurveRefit = ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_CURVE_REFIT"] == "1"
    var curveDbgPrinted = false   // VIZ_CURVE_DEBUG one-shot
    var rtInstUniformBuffer: MTLBuffer!
    var rtInstFrameSeed: UInt32 = 0
    /// DH-0887 — +1 per TLAS lighting dispatch, never by any other pass, so the GI's Sobol index
    /// is contiguous (`rtInstFrameSeed` also advances in the glass pass, and every-other Sobol
    /// index covers only half of the unit square).
    var rtGIProgressiveIndex: UInt32 = 0

    /// True after a successful TLAS (re)build/refit this frame — the RT pass
    /// then traces the instance AS instead of the primitive-AS soup.
    var rtTLASActive: Bool = false

    // ── Phase 4.37a — RT hang-proof guard ────────────────────────────────
    // A scene whose geometry is GPU-fed (regenerated `IlluminatoramaMesh`
    // identities each frame) or whose mesh set/topology otherwise changes
    // every frame makes `rtTopologyHash()` differ every frame → `updateRTAccel`
    // takes the REBUILD branch each frame (new per-mesh BLAS builds, each with
    // a `waitUntilCompleted`, + a TLAS rebuild) → the main thread hangs and a
    // faulted AS traces to magenta. The descriptor-driven per-scene RT opt-in
    // (Phase 4.36/4.37) runs RT in the live tick loop, so any flagged scene
    // could trip this. Two gates make RT fail safe (fall back to non-RT, never
    // hang): an UP-FRONT size cap, and a rebuild-THRASH counter.
    //
    // The guard used to LATCH until the next scene attach, and that was wrong
    // for an editor. In a document app the model mutates on every tick of a
    // drag (push/pull a wall ⇒ new wall meshes ⇒ a new topology hash every
    // frame), so seven frames into the drag the thrash counter tripped, RT
    // glass switched off mid-gesture, and it NEVER came back: a perf spike
    // followed by a permanent visual pop, with no user-reachable way to restore
    // it short of reopening the document. The guard's real job is "don't rebuild
    // a thrashing TLAS", which stops being true the moment the scene settles.
    //
    // So the trip is now RECOVERABLE. While disabled the renderer keeps
    // evaluating cheap evidence each frame — the size caps (a scene can shrink
    // back under them) and the topology hash — and re-arms RT once the topology
    // has held still for `rtRecoveryStableFrames`. A genuinely thrashing scene
    // (Forest: GPU-fed leaf geometry, a different hash every single frame) never
    // satisfies that and stays disabled exactly as before, which is the
    // property the hang-proofing depends on.
    /// The guard's whole state, as one testable value (see
    /// `IlluminatoramaRTHangGuard` for the recovery semantics and why they
    /// changed from a permanent latch).
    var rtGuard = IlluminatoramaRTHangGuard()

    var rtConsecutiveRebuilds: Int = 0
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// **Daydream DH-1320 — which of a host's room-fill STAND-INS the one-bounce traced GI owns.**
    /// A host that lights interiors without transport (interior irradiance bands, an ambient
    /// supplement, window apertures as area lights) is describing the light a traced bounce
    /// brings back: the bounce's misses ARE the sky through the window and its hits are lit
    /// surfaces. With the traced GI running, each named stand-in is handed to it the way
    /// `rtGIReplacesDiffuseSky` hands over the outdoor-cube sky (the lighting pass writes the share
    /// to the hand-off target, the traced pass subtracts it where its rays ran) — the one-bounce
    /// sibling of `rtPathTracedGI`'s whole-indirect hand-off, one bit per stand-in so each can be
    /// measured alone. Needs `rtGIReplacesDiffuseSky`; ignored on the path lane (which already
    /// owns all of it). Empty (default) ⇒ byte-identical.
    public struct GIRoomFillHandoff: OptionSet, Sendable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }
        /// The whole diffuse IBL — interior bands, room gains, interior scalars (frame bit 16).
        public static let bands = GIRoomFillHandoff(rawValue: 1)
        /// The ambient supplement (frame bit 32).
        public static let ambient = GIRoomFillHandoff(rawValue: 2)
        /// Area lights flagged `isAperture` — window portals (frame bit 64).
        public static let apertures = GIRoomFillHandoff(rawValue: 4)
        public static let all: GIRoomFillHandoff = [.bands, .ambient, .apertures]
    }

    // ── Who computes the sun's DIRECT term (C1) ──────────────────────
    //
    // THE BUG THIS EXISTS TO MAKE IMPOSSIBLE: the RT lighting kernels end with
    // `outHDR += direct + indirect + reflection`, and `direct` was computed
    // unconditionally whenever `shadowRays > 0` (default 4). The frame graph never
    // suppressed the DEFERRED lighting pass — it only chose *which* RT pass ran —
    // so a host that flipped `rtEnabled` on a normally-lit scene got the sun
    // TWICE, on top of everything else. Every Visualizer host avoided that by
    // CONVENTION (each one zeroes `directionalLightColor` + `shadowsEnabled`
    // itself, and `IlluminatoramaSceneExtractor` does it for the descriptor path);
    // a convention is exactly what a new host forgets. Daydream Home forgot it, and
    // "RT GI washes interiors out" was mostly this.
    //
    // Ownership is now a stated property of the renderer, and the RT kernels are
    // told the answer in a uniform (`directSunEnabled`) instead of assuming it.
    public enum RTSunOwnership: Sendable {
        /// The DEFERRED lighting pass owns the sun; the RT pass contributes only
        /// GI + reflections. In this mode `rtGIStrength = 0` + `rtReflStrength = 0`
        /// makes an RT-on frame BIT-IDENTICAL to an RT-off frame — the property
        /// `IlluminatoramaRTSunOwnershipTests` gates, and the cheapest possible
        /// alarm on this whole class of bug.
        ///
        /// This is the right mode when the deferred BRDF is richer than the RT
        /// one (it is: cascades, clearcoat, sheen, anisotropy, transmission) and
        /// RT is wanted for transport, not for the sun.
        case deferred
        /// The RT pass owns the sun — soft shadow rays sampled across the sun's
        /// angular disc, i.e. a true penumbra the cascades can't produce. The
        /// renderer FORCES the deferred directional light + cascaded shadows off
        /// for the frame, so choosing this can't leave the sun double-counted.
        case rayTraced
        /// Resolve from observable state: `rayTraced` iff the host has already
        /// zeroed `directionalLightColor` (the hand-over every RT host in
        /// Visualizer performs), else `deferred`. This is the default because it
        /// reproduces EXACTLY what every existing host does today — the scenes
        /// that hand the sun over keep RT's sun, and a host that never did (i.e.
        /// was double-counting) stops.
        case automatic
    }

    // ── TLAS instance masks (C3) ─────────────────────────────────────
    //
    // Three states, not two. The first two are old:
    //   0x01  traced — an ordinary opaque/curve instance.
    //   0x02  AAA glass — a pane, so it never casts a solid shadow.
    //   0x00  excluded — present in the TLAS layout, never intersected. The RT
    //         representation lives elsewhere (a superquadric's proxy, a curve twin).
    // and the third is new:
    //   0x04  INVISIBLE OCCLUDER — real to LIGHT, never drawn. Daydream Home's
    //         lighting-only `ceilshadow.*` ceilings are the case: a roofless
    //         dollhouse needs a ceiling that stops the sun (it is already a
    //         shadow-map occluder) but must not appear in the picture. With only
    //         the binary states available, the bridge had to pick "excluded", and
    //         then every GI ray leaving an interior surface through the open top
    //         missed everything and returned FULL SKY — the patio flood
    //         `+InteriorLight` exists to prevent.
    //
    // The split is by ray PURPOSE, not by pass: transport rays (shadow, GI) trace
    // `0x01|0x04`; camera-visible rays (glass primary/refraction, glossy
    // reflections) trace without 0x04, so an undrawn slab is never shown.
    static let rtMaskTraced: UInt32 = 0x01

    static let rtMaskGlass: UInt32 = 0x02

    static let rtMaskInvisibleOccluder: UInt32 = 0x04

    /// Shadow + GI ray mask: opaque/curve + invisible occluders.
    static let rtTransportRayMask: UInt32 = rtMaskTraced | rtMaskInvisibleOccluder

    /// Restart the RT GI accumulation on the next frame (its history reprojects against nothing),
    /// WITHOUT `resetTemporalHistory`'s exposure re-seed. For a host whose still discards warm-up
    /// frames: in `rtGIStaticAccumulation` mode nothing forgets them otherwise (DH-0887).
    /// `keepingRadianceCache` — restart the GI running mean but keep the path lane's radiance cache (a
    /// still that has just BUILT it in held-out frames, DH-1344).
    public func resetRTGIAccumulation(keepingRadianceCache: Bool = false) {
        let rcClear = rts.rcNeedsClear
        rtGINeedsFirstFrame = true
        if keepingRadianceCache { rts.rcNeedsClear = rcClear }
    }

    /// Diagnostic A/B override: `VIZ_ILLUMI_SVGF=1`/`0` forces the SVGF cascade
    /// on/off for every Illuminatorama renderer in the process, so a headless
    /// render can exercise the à-trous path (issue #65) without touching the
    /// default. `nil` when unset. Read once. `effectiveSVGFEnabled` consults it.
    static let svgfEnvOverride: Bool? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF"] else { return nil }
        return v == "1" || v.lowercased() == "true"
    }()

    /// Diagnostic A/B override for `svgfLumFloor` (`VIZ_ILLUMI_SVGF_LUMFLOOR`).
    static let svgfLumFloorEnv: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF_LUMFLOOR"],
              let f = Float(v) else { return nil }
        return f
    }()

    /// Diagnostic A/B override for cascade level count (`VIZ_ILLUMI_SVGF_LEVELS`).
    static let svgfLevelsEnv: Int? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF_LEVELS"],
              let i = Int(v) else { return nil }
        return i
    }()

    // ── Hardware ray tracing state ───────────────────────────────────
    /// Mirror of `RTUniforms` in IlluminatoramaRT.metal.
    struct RTUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var _pad0: Float = 0
        var sunDir: SIMD3<Float>;         var sunSoftnessRad: Float
        var sunColor: SIMD3<Float>;       var giStrength: Float
        var skyAmbient: SIMD3<Float>;     var specStrength: Float
        var width: UInt32; var height: UInt32
        var shadowRays: UInt32; var giRays: UInt32
        var frameSeed: UInt32; var rayTMin: Float; var maxGIDist: Float; var triangleCount: UInt32
        var reflStrength: Float; var reflMaxDist: Float; var reflRoughnessCutoff: Float
        var reflRays: UInt32; var reflEnabled: UInt32
        // Surface cache: read cached radiance at GI/reflection hits when enabled.
        var surfCacheEnabled: UInt32; var surfTileSize: UInt32; var surfTilesPerRow: UInt32
        var surfAtlasW: UInt32; var surfAtlasH: UInt32
        var _padEmitters: UInt32 = 0
        var leafTransmission: Float = 0      // mirrors _padRT0 in the Metal struct
        // Curve primitives (#60 item 7, incr. 2): 1 ⇒ soup AS holds a curve
        // geometry descriptor; mirrors `curvesEnabled` (was `_padRT1`) in Metal.
        var curvesEnabled: UInt32 = 0
        /// C1 — 0 ⇒ the deferred pass already shaded the sun here (mirrors
        /// `directSunEnabled`, was `_padRT2`). See `RTSunOwnership`.
        var directSunEnabled: UInt32 = 0
        /// DH-0872 — scotopic desaturation for the GI sky-miss sample only. Mirrors
        /// `scotopicDesaturation` (the renderer's own property, also fed to the
        /// tonemap pass). 0 (day) ⇒ byte-identical.
        var scotopicDesaturation: Float = 0
    }

    /// Mirror of `RTDenoiseUniforms` in Illuminatorama.metal (stride 32, 16-aligned).
    struct RTDenoiseUniforms {
        var width: UInt32; var height: UInt32; var enabled: UInt32; var radius: UInt32
        var kDepth: Float; var kNorm: Float; var _pad0: Float = 0; var _pad1: Float = 0
    }

    /// Mirror of `RTGITemporalUniforms` in Illuminatorama.metal (stride 32, 16-aligned).
    struct RTGITemporalUniforms {
        var width: UInt32; var height: UInt32; var enabled: UInt32; var isFirstFrame: UInt32
        var blend: Float; var gammaClamp: Float; var staticAccumulation: UInt32 = 0; var _pad1: Float = 0
    }

    // ── TLAS — instanced RT for animated extracted scenes ────────────
    // Per-mesh BLAS built once (object space); a TLAS of per-instance
    // transforms refit each frame. Replaces the room-only / soup primitive-AS
    // path for extracted overlay scenes so RT generalises to ANIMATION without
    // a per-frame vertex transform + geometry rebuild.
    /// Mirror of `RTInstanceData` in IlluminatoramaRTInstanced.metal.
    struct RTInstanceData {
        var nrm0: SIMD4<Float>; var nrm1: SIMD4<Float>; var nrm2: SIMD4<Float>
        var albedoTriBase: SIMD4<Float>   // xyz = albedo, w = objNormal base (as Float)
        /// xyz = the instance's SELF-LIT radiance (the emission scalar, intensity
        /// already folded in). A lamp shade, a TV panel and the night window glow are
        /// all lit by this term and by nothing else, so a secondary ray that omits it
        /// renders every one of them dim or absent — seen through a pane, the lamp
        /// lighting the room went missing. Read by `secondaryEmission` in
        /// IlluminatoramaSecondary.h. w: GLASS rows only — first corner-normal entry + 1 in
        /// `rtGlassCornerNormalBuffer` (0 = none); opaque rows keep it 0.
        var emissionPad: SIMD4<Float> = .zero
    }

    /// Mirror of `RTInstUniforms`.
    struct RTInstUniforms {
        var invViewProjection: simd_float4x4
        /// 1 ⇒ the traced GI replaces the deferred diffuse sky (`rtGIReplacesDiffuseSky`). Was `_pad0`.
        var cameraWorldPos: SIMD3<Float>; var giReplacesDiffuseSky: UInt32 = 0
        var sunDir: SIMD3<Float>; var sunSoftnessRad: Float
        var sunColor: SIMD3<Float>; var giStrength: Float
        var skyAmbient: SIMD3<Float>; var specStrength: Float
        var width: UInt32; var height: UInt32
        var shadowRays: UInt32; var giRays: UInt32
        var frameSeed: UInt32; var rayTMin: Float; var maxGIDist: Float
        /// DH-0896 — 1 ⇒ `specIBLTexture` holds this frame's deferred specular-IBL share and
        /// a reflection HIT replaces it. Was `_pad1` (same 4 bytes, stride unchanged).
        var reflReplacesIBL: UInt32 = 0
        var reflStrength: Float; var reflMaxDist: Float; var reflRoughnessCutoff: Float
        var reflRays: UInt32; var reflEnabled: UInt32
        // Surface cache (P1c) — mirror of the Metal RTInstUniforms tail.
        var surfCacheEnabled: UInt32 = 0
        var surfTileSize: UInt32 = 1; var surfTilesPerRow: UInt32 = 1
        var surfAtlasW: UInt32 = 1; var surfAtlasH: UInt32 = 1
        var surfTriCount: UInt32 = 0
        var debugSurfCacheGI: UInt32 = 0
        // Curve primitives (#60 item 7): TLAS instance ids ≥ curveInstanceBase
        // are curve sets (id − base = set index). 0 sets ⇒ fields unread.
        var curveInstanceBase: UInt32 = 0
        var curveSetCount: UInt32 = 0
        // Phase 5 / B0 — DebugTerm.surfaceCacheVariance isolation (was _padc0).
        var debugSurfCacheVar: UInt32 = 0
        // Phase 5 / A — residency feedback gate (was _padc1).
        var surfFeedbackEnabled: UInt32 = 0
        // ── Secondary-hit shading parity (shared with the AAA glass pass) ────
        // Feed `SecondaryShadeParams` in IlluminatoramaSecondary.h. Defaults are
        // off/neutral, so a host that wires none of them renders as before.
        var skyIntensity: Float = 1
        var interiorMask: UInt32 = 0
        var interiorIBLUp: Float = 1
        var interiorIBLSide: Float = 1
        var interiorAmbient: Float = 1
        var albedoAtlasEnabled: UInt32 = 0
        var objUVCount: UInt32 = 0
        var pointLightCount: UInt32 = 0
        var spotLightCount: UInt32 = 0
        // C1 — 0 ⇒ the deferred pass already shaded the sun here; this kernel must
        // not add it again (see `RTSunOwnership`). C3 — the shadow/GI ray mask,
        // 0x01 opaque | 0x04 invisible occluder.
        var directSunEnabled: UInt32 = 0
        var transportRayMask: UInt32 = 0x05
        // DH-0653 — hit/miss counter gate + chart overlay (were _padIrr0/_padIrr1).
        var surfStatsEnabled: UInt32 = 0; var debugSurfCacheCharts: UInt32 = 0
        /// DH-0872 — scotopic desaturation for the GI sky-miss sample only. Mirrors
        /// `scotopicDesaturation` (also fed to the tonemap pass). Repurposes
        /// `_padIrr2` — same 4 bytes. 0 (day) ⇒ byte-identical.
        var scotopicDesaturation: Float = 0
        // Interior irradiance bands — mirror of the Metal RTInstUniforms tail.
        var interiorIrrUp: SIMD4<Float> = .zero
        var interiorIrrSide: SIMD4<Float> = .zero
        var interiorIrrDown: SIMD4<Float> = .zero
        // Per-room band level (S3.5 Stage E) — so a room in a reflection is scaled
        // like the room seen directly. Same packing as the frame twin.
        var interiorRoomGain0: SIMD4<Float> = .one
        var interiorRoomGain1: SIMD4<Float> = .one
        var interiorRoomGain2: SIMD4<Float> = .one
        var interiorRoomGain3: SIMD4<Float> = .one
        var interiorRoomGain4: SIMD4<Float> = .one
        var interiorRoomGain5: SIMD4<Float> = .one
        var interiorRoomGain6: SIMD4<Float> = .one
        var interiorRoomGain7: SIMD4<Float> = .one
        var interiorRoomGainMeta: SIMD4<Float> = .zero
        // DH-0887 — the GI directions' contiguous progressive index (Metal twin appended last).
        var giProgressiveIndex: UInt32 = 0
        /// DH-0715 — the traced-origin snap's reach (was `padProg0`, same 4 bytes).
        var surfaceSnapULPs: Float = 0
        /// DH-0989 — path bounces (0 ⇒ one-bounce GI) and the per-path luminance cap.
        var pathBounces: UInt32 = 0
        var pathClamp: Float = 0
        /// DH-1011 — `kPathFlag*` (Metal twin appended last; three pads keep the 16-byte tail).
        var pathFlags: UInt32 = 0
        /// DH-1293 — the lamps' near field at path vertices (were `padPath0/padPath1`, same bytes).
        var pathLampMinDist: Float = 0, pathLampSourceRadius: Float = 0
        /// DH-1344 — the path lane's radiance-cache cell (m); 0 ⇒ off (was `padPath2`).
        var radCacheCell: Float = 0
        /// DH-1308 — the glossy vertices' roughness floor (Metal twin appended last; three pads keep the
        /// 16-byte tail).
        var pathGlossyMinRoughness: Float = 0
        var pathGlossyLampSize: Float = 1, pathGlossyLampMinRoughness: Float = 0, padGlossy2: Float = 0

        mutating func setInteriorRoomGains(_ gains: [Float], enabled: Bool) {
            let p = InteriorRoomGains.pack(gains, enabled: enabled)
            interiorRoomGain0 = p.v[0]; interiorRoomGain1 = p.v[1]
            interiorRoomGain2 = p.v[2]; interiorRoomGain3 = p.v[3]
            interiorRoomGain4 = p.v[4]; interiorRoomGain5 = p.v[5]
            interiorRoomGain6 = p.v[6]; interiorRoomGain7 = p.v[7]
            interiorRoomGainMeta = p.meta
        }
    }

    /// **The ONE place an instance becomes `RTInstanceData`.** The normal matrix's
    /// column `.w` lanes are dead weight for RT (every consumer builds a float3x3
    /// from `.xyz`), so they carry the two facts a re-shaded secondary hit needs to
    /// shade like the deferred pass does:
    ///   nrm0.w = albedo atlas slice PLUS ONE — so the memset-0 default (curve
    ///            instances, unslotted instances) decodes to −1 = "untextured, use
    ///            the mean albedo" rather than slice 0
    ///   nrm1.w = the instance's light-layer bits (bit-cast; drives light masking +
    ///            the interior day-light separation)
    /// plus `emissionPad`, the self-lit radiance.
    ///
    /// This exists because the per-frame REFIT used to write `dataPtr[i].nrm0 =
    /// nm.columns.0` directly — which wipes both packed lanes, so from the second
    /// frame after a rebuild every secondary hit silently lost its texture and its
    /// light masking, and (once emission joined the struct) would have kept a stale
    /// glow while the real one ramped with nightfall. Rebuild and refit must produce
    /// byte-identical rows; the only way to guarantee that is to have one function.
    ///
    /// `normalBase` is the instance's offset into the concatenated `objNormal` /
    /// `objUV`, which is topology (fixed at rebuild), so the refit passes back the
    /// value already in the row.
    static func rtInstanceRow(_ inst: IlluminatoramaInstance, normalBase: Float) -> RTInstanceData {
        var c0 = inst.normalMatrix.columns.0, c1 = inst.normalMatrix.columns.1
        var c2 = inst.normalMatrix.columns.2
        c0.w = Float(inst.albedoTextureSlice + 1)
        c1.w = Float(bitPattern: inst.layer)
        // DH-1308 — the layered stack a path vertex samples (`secondaryMaterial` / `secondaryCoat`):
        // the base's metalness + roughness in `emissionPad.w`, the clearcoat's strength + roughness in
        // `nrm2.w` (0 strength ⇒ no coat). Read only under `kPathFlagGlossyVertices`.
        c2.w = inst.clearcoat > 0 ? rtPackMaterialLane(inst.clearcoat, inst.clearcoatRoughness) : 0
        return RTInstanceData(nrm0: c0, nrm1: c1, nrm2: c2,
                              albedoTriBase: SIMD4(inst.albedo, normalBase),
                              emissionPad: SIMD4(inst.emission, rtPackMaterialLane(inst.metallic, inst.roughness)))
    }

    /// DH-1308 — two [0, 1] material scalars in ONE float lane, `−(1 + 2·round(a·1000) + min(b, 0.999))`:
    /// always ≤ −1, so a glass row's corner-normal base (≥ 0) and an unpacked row's 0 both decode to
    /// "absent" (`secondaryUnpackMaterialLane`, IlluminatoramaSecondary.h). `a` keeps 1/1000, `b` ~1e-4.
    nonisolated static func rtPackMaterialLane(_ a: Float, _ b: Float) -> Float {
        let aq = (min(max(a, 0), 1) * 1000).rounded()
        return -(1 + 2 * aq + min(max(b, 0), 0.999))
    }

    /// Glass sibling of `rtInstanceRow`. A dielectric boundary is never a shaded hit
    /// surface — the bounce loop reads its material from `rtGlassDataBuffer` — so the
    /// packed lanes are pinned to explicit "untextured / every layer / no emission"
    /// values instead of being left to whatever the normal matrix's padding holds.
    static func rtGlassInstanceRow(_ inst: IlluminatoramaGlassInstance, normalBase: Float,
                                           cornerBase: Float = 0) -> RTInstanceData {
        var c0 = inst.normalMatrix.columns.0, c1 = inst.normalMatrix.columns.1
        c0.w = 0                                    // untextured
        c1.w = Float(bitPattern: UInt32.max)        // unmasked by any light layer
        return RTInstanceData(nrm0: c0, nrm1: c1, nrm2: inst.normalMatrix.columns.2,
                              albedoTriBase: SIMD4(inst.tintIor.x, inst.tintIor.y, inst.tintIor.z, normalBase),
                              // w = first corner-normal entry + 1 (0 = none); a glass boundary
                              // has no emission, so the xyz stay zero.
                              emissionPad: SIMD4(0, 0, 0, cornerBase))
    }

    /// Everything the RT path caches per registered mesh: its BLAS, its object-space
    /// per-triangle normals, and — for a deforming (GPU-fed) mesh — the descriptor and
    /// scratch its per-frame in-place refit needs. ONE entry, not four parallel maps, so a
    /// mesh's RT state can only ever be added, found and dropped as a unit.
    ///
    /// **The entry OWNS its mesh, and that is load-bearing.** The cache is keyed by
    /// `ObjectIdentifier(mesh)` — a raw ADDRESS. A document editor rebuilds its whole scene
    /// on every edit (`HouseRenderBridge.load()` drops the previous scene's mesh handles and
    /// registers new ones), and `IlluminatoramaMeshHandle.deinit` evicts the renderer entry
    /// ASYNCHRONOUSLY (`Task { @MainActor … removeMesh }`). So without a strong reference the
    /// key dangles the moment that task runs, and the very next `IlluminatoramaMesh` the
    /// allocator hands out can land on the recycled address — at which point the cache
    /// "hits" and gives brand-new geometry an acceleration structure built from a mesh it
    /// has never seen, still referencing the dead mesh's released vertex/index buffers.
    /// Holding the mesh pins the address for exactly as long as the entry exists;
    /// `pruneRTMeshCache()` is the only thing that lets go.
    struct RTMeshEntry {
        /// STRONG on purpose — see above. Never make this `weak`/`unowned`.
        let mesh: IlluminatoramaMesh
        let blas: MTLAccelerationStructure
        let normals: [SIMD4<Float>]
        /// Per-triangle mesh UVs (3 per triangle), in the SAME triangle order as
        /// `normals` — so one `normalBase + primitive_id` resolves both a hit's
        /// world normal and its texture coordinates. Feeds `rtObjUVBuffer`.
        let uvs: [SIMD2<Float>]
        // Phase B — deforming-mesh BLAS refit. A DynamicMesh / GPU-fed mesh keeps a
        // STABLE topology (vertex+index count fixed; only positions move each frame),
        // so its BLAS is built once with `.refit` usage and re-fit per frame from the
        // live (repacked) vertex buffer — instead of being frozen at build-time verts,
        // which made the RT trace see stale geometry (shadows/GI lit for the old pose).
        // The refit is encoded on the FRAME command buffer AFTER `encodeGPURepacks`, so
        // Metal sequences repack-write → refit-read; the BLAS references `mesh.vertexBuffer`,
        // the same buffer the repack writes in place. `nil` for static meshes.
        var refit: (desc: MTLPrimitiveAccelerationStructureDescriptor, scratch: MTLBuffer)?
        /// Per-triangle-CORNER shading normals (3 per triangle, `normals` order) — filled only
        /// for a mesh some glass instance uses, and read only by the glass bounce loop, where a
        /// dielectric boundary's normal is its refraction (`objectCornerNormals`). Empty for
        /// every opaque-only mesh, so the scene pays nothing for it. Feeds
        /// `rtGlassCornerNormalBuffer`.
        var cornerNormals: [SIMD4<Float>] = []
    }

    /// Drop every cached entry whose mesh is no longer registered, releasing its BLAS, its
    /// normals and — the point of the whole exercise — the mesh itself, so the address it
    /// occupied becomes reusable only once nothing can look it up any more.
    ///
    /// Called from `rebuildRTAccel()`, i.e. exactly when the TLAS topology changed, which is
    /// the only moment the set of live meshes can have moved.
    func pruneRTMeshCache() {
        // Filtered unconditionally: equal counts do NOT imply equal sets (a scene can shed
        // one mesh and gain another in the same rebuild), and this runs on topology changes
        // only, never per frame.
        let live = Set(meshes.values.map(ObjectIdentifier.init))
        rts.rtMeshCache = rts.rtMeshCache.filter { live.contains($0.key) }
        rts.rtHasDeformingBLAS = rts.rtMeshCache.values.contains { $0.refit != nil }
    }

    /// Mirror of `RTCurveSetData` in IlluminatoramaRTInstanced.metal (112 B,
    /// SIMD4-aligned — no SIMD3 trap). m0..m3 = the set's object→world matrix
    /// columns; meta.x = the set's first segment index in the pooled buffers.
    struct RTCurveSetData {
        var m0: SIMD4<Float>; var m1: SIMD4<Float>; var m2: SIMD4<Float>; var m3: SIMD4<Float>
        var albedoRoughness: SIMD4<Float>
        var emissionPad: SIMD4<Float>
        var meta: SIMD4<UInt32>
    }

    /// Mirror of `CurveWindUniforms` in IlluminatoramaRT.metal (16 B).
    struct CurveWindUniforms {
        var time: Float; var windStrength: Float; var windHeading: Float; var pointCount: UInt32
    }

    /// Rebuild this many frames running ⇒ topology is thrashing ⇒ disable RT.
    /// Allows a few legitimate settle-frames at scene load before giving up.
    /// (Backup gate — the size cap below is the primary defence, because the
    /// thrash counter can't catch a SINGLE catastrophic rebuild that never
    /// returns.)
    static let rtRebuildThrashLimit: Int = 6

    /// Scenes past ANY of these caps are too heavy for per-frame RT in the live
    /// tick loop (a single rebuild — thousands of BLAS + CPU normal readbacks +
    /// `waitUntilCompleted` — would stall for seconds-to-minutes). Checked UP
    /// FRONT, before the first build, so a heavy scene auto-disables instead of
    /// hanging. The triangle cap is the real cost predictor (catches both
    /// many-small-mesh and few-huge-mesh scenes); the instance / mesh-group
    /// caps catch BLAS/TLAS build fan-out. Tuned for "light enough that a
    /// per-frame rebuild stays sub-frame" — the room / a box scene pass; Forest
    /// (thousands of leaf cards) does not.
    static let rtMaxInstancesForLiveRT: Int = 2048

    static let rtMaxMeshGroupsForLiveRT: Int = 128

    static let rtMaxTrianglesForLiveRT: Int = 150_000

    /// Triangle cap for the GLASS-ONLY TLAS (AAA glass opted in, but full-scene RT
    /// lighting is OFF). The strict 150k cap above is sized so a per-pixel RT
    /// lighting/GI pass stays sub-frame — but a glass-only TLAS is traced by ONLY
    /// the handful of glass fragments in frame, so the same opaque geometry is far
    /// cheaper to carry. An architectural interior (walls/floor/ceiling/furniture +
    /// a lawn's worth of ground triangles) routinely lands in the 150k–600k range;
    /// this higher ceiling lets its windows refract the real room without tripping
    /// the hang-guard. Still bounded so a pathological scene auto-disables rather
    /// than stalling the first BLAS build. Only consulted on the glass-only path.
    ///
    /// **2.5 M is measured; the old 1.2 M was not** (Daydream DH-1367, `DocumentRTCensusTests
    /// .testRTBuildCostVersusTriangleCount`: the real 4000 Sunset (north side) document — which
    /// sat at 1.247 M and lost every RT pass, its mirrors showing the sky — padded with distinct
    /// ~250 k-triangle meshes, live lane, 1600×1000, Debug):
    ///
    ///     unique tris │ 1.25 M  1.50 M  2.00 M  2.49 M  3.24 M
    ///     rebuild     │   75     114     193     170     240  ms
    ///     steady      │   92.1   90.3    85.9    94.6    90.5 ms
    ///     GPU alloc   │ 4879    4931    5034    5137    5291  MB
    ///
    /// The steady frame does not move (BVH traversal is logarithmic); the rebuild frame grows
    /// ~8 ms per 100 k triangles and memory ~0.2 KB per triangle. A rebuild runs only on a
    /// TOPOLOGY change, which a host's settled lane sees once per committed edit. Nowhere on
    /// the ladder is the seconds-to-minutes stall this guard exists to prevent; 2.5 M is ~2× the
    /// heaviest real document and keeps the worst rebuild near 200 ms.
    static let rtMaxTrianglesForGlassOnlyRT: Int = 2_500_000

    /// Mesh-GROUP cap for the GLASS-ONLY TLAS — the exact asymmetry the triangle caps
    /// above already encode (150 k live vs 1.2 M glass-only), applied to the other axis.
    ///
    /// **Why it had to move.** 128 is a *live-RT* number: there, a topology change can
    /// mean a full rebuild every frame in the tick loop, so the cap has to keep one
    /// rebuild well inside a frame. The glass-only path rebuilds only on a real topology
    /// change (the thrash guard below still catches per-frame churn) and hosts run it in a
    /// settled/photo lane, so it can carry far more. Meanwhile 128 is *small* for real
    /// content: an architectural document's group count is "how many distinct geometries",
    /// which scales with rooms and cabinetry — one group per wall edge, per room floor,
    /// per millwork sub-part. A 2-storey furnished house measures 209 and every pane in it
    /// silently dropped to the flat Fresnel fallback, scene-wide, on exactly the class of
    /// scene AAA glass exists for.
    ///
    /// **512 is measured, not guessed** (Daydream `DocumentRTCensusTests
    /// .testRTGlassBuildCostVersusMeshGroupCount`: a real document padded with N extra
    /// distinct 12-triangle groups, so group count is the only variable moving):
    ///
    ///     groups │  209    309    509    909   1509
    ///     rebuild│ 21.2   29.1   34.5   46.2   60.2  ms
    ///     steady │ 22.4   21.6   23.7   29.1   30.5  ms
    ///
    /// **There is no cliff on this axis** — the rebuild frame is linear at ≈0.030 ms per
    /// group out to 3× this cap, so the failure mode a cap must prevent (a first build that
    /// never returns) is a triangle/instance phenomenon, not a group one. At 512 the
    /// worst-case rebuild frame is ~34 ms: a hitch in a lane that has no vsync budget, and
    /// nowhere near the seconds-to-minutes stall the guard exists for. The cap stays finite
    /// so a pathological scene still auto-disables rather than degrading unboundedly.
    ///
    /// The live-RT path keeps 128, untouched — Visualizer's live-RT scenes are unaffected.
    static let rtMaxMeshGroupsForGlassOnlyRT: Int = 512

    /// Reset the RT auto-disable guard. Called on every scene attach (via
    /// `IlluminatoramaOverlay.setExtractedSceneRT`) so a freshly-shown scene
    /// gets a clean chance at RT regardless of what the previous scene did.
    public func resetRTGuard() {
        rts.rtGuard.reset()
        rts.rtConsecutiveRebuilds = 0
    }

    // ── Hardware RT geometry ──────────────────────────────────────────
    //
    // Supply the static world-space triangle soup the RT pass traces against
    // (the room's walls / floor / ceiling, with the window left as a hole so
    // sun rays pass through). Per-triangle albedo + geometric normal travel in
    // parallel arrays so the GI bounce can shade hit points. Builds the
    // primitive acceleration structure once; call again only if the geometry
    // changes. No-op on hardware without RT support.
    /// Diagnostic: append an AS-build command buffer's error to the sidecar file
    /// named by `VIZ_ILLUMI_CBERROR_PATH` (sandbox eats os_log). No-op without it.
    nonisolated static func recordASBuildError(_ cmd: MTLCommandBuffer, _ label: String) {
        guard cmd.status == .error || cmd.error != nil,
              let p = ProcessInfo.processInfo.environment["VIZ_ILLUMI_CBERROR_PATH"] else { return }
        let nsErr = cmd.error as NSError?
        let line = "\(label) status=\(cmd.status.rawValue) domain=\(nsErr?.domain ?? "?") code=\(nsErr?.code ?? 0) \(nsErr?.localizedDescription ?? "")\n"
        guard let data = line.data(using: .utf8) else { return }
        if let h = FileHandle(forWritingAtPath: p) { h.seekToEndOfFile(); h.write(data); try? h.close() }
        else { try? data.write(to: URL(fileURLWithPath: p)) }
    }

    public func setRTGeometry(positions: [SIMD3<Float>],
                              indices: [UInt32],
                              triangleAlbedo: [SIMD3<Float>],
                              triangleNormal: [SIMD3<Float>]) {
        guard rtSupported, rts.rtPipeline != nil else { return }
        guard !positions.isEmpty, indices.count % 3 == 0, !indices.isEmpty else { return }
        let triCount = indices.count / 3
        guard triangleAlbedo.count == triCount, triangleNormal.count == triCount else {
            Self.log.error("setRTGeometry: per-triangle arrays must match triangle count")
            return
        }
        // Enforce the index↔vertex invariant at this registration boundary too
        // (symmetric with IlluminatoramaMesh.sanitiseIndices on the from() path):
        // an index >= positions.count would make the GPU acceleration-structure
        // BUILD read vertices past the buffer — a GPU fault that's far harder to
        // diagnose than this early return (the app sandbox eats os_log, so a fault
        // shows up only as a dropped frame / device loss). Callers are controlled
        // today, so this is defense-in-depth, not a known bug.
        // See docs/known-issues/illuminatorama-index-exceeds-vertexcount.md.
        if let mx = indices.max(), Int(mx) >= positions.count {
            Self.log.error("setRTGeometry: index \(mx) >= vertexCount \(positions.count); rejecting (would fault the AS build).")
            return
        }
        // Vertex positions as packed float3 (12-byte) for the AS triangle
        // descriptor; index buffer as uint32; per-tri albedo/normal as float4.
        var packedPos = [SIMD3<Float>]()  // 16-byte stride; AS reads .float3 with stride 16
        packedPos.reserveCapacity(positions.count)
        packedPos.append(contentsOf: positions)
        let albedo4 = triangleAlbedo.map { SIMD4<Float>($0, 1) }
        let normal4 = triangleNormal.map { SIMD4<Float>(simd_normalize($0), 0) }

        guard
            let vb = device.makeBuffer(bytes: packedPos,
                length: MemoryLayout<SIMD3<Float>>.stride * packedPos.count,
                options: .storageModeShared),
            let ib = device.makeBuffer(bytes: indices,
                length: MemoryLayout<UInt32>.stride * indices.count,
                options: .storageModeShared),
            let ab = device.makeBuffer(bytes: albedo4,
                length: MemoryLayout<SIMD4<Float>>.stride * albedo4.count,
                options: .storageModeShared),
            let nb = device.makeBuffer(bytes: normal4,
                length: MemoryLayout<SIMD4<Float>>.stride * normal4.count,
                options: .storageModeShared)
        else {
            Self.log.error("setRTGeometry: buffer allocation failed")
            return
        }
        vb.label = "Illuminatorama.rt.positions"
        ib.label = "Illuminatorama.rt.indices"
        ab.label = "Illuminatorama.rt.triAlbedo"
        nb.label = "Illuminatorama.rt.triNormal"
        rts.rtVertexBuffer = vb
        rts.rtIndexBuffer = ib
        rts.rtTriAlbedoBuffer = ab
        rts.rtTriNormalBuffer = nb
        rts.rtTriangleCount = triCount

        buildSoupAccel()   // builds the primitive AS (triangle soup + curve geom when active)
        Self.log.notice("RT geometry built: \(triCount) triangles")
    }

    /// Build the soup primitive acceleration structure from the stored triangle
    /// buffers, plus — when a soup curve set is active (#60 item 7 incr. 2) — a
    /// SECOND curve geometry descriptor (round Catmull-Rom wood). Rebuilt on a
    /// topology change (new triangle geometry OR a curve registry version bump).
    /// When curves are active the AS uses `.refit` usage so the per-frame sway
    /// kernel + `refitRTSoupAccel` can update the control points in place.
    func buildSoupAccel() {
        guard let vb = rts.rtVertexBuffer, let ib = rts.rtIndexBuffer, rts.rtTriangleCount > 0 else { return }
        // Triangle soup AS — its OWN primitive AS (curves can't share it). Static,
        // so default usage (the room/extractor scenes never refit this).
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
        geom.vertexBuffer = vb
        geom.vertexBufferOffset = 0
        geom.vertexStride = MemoryLayout<SIMD3<Float>>.stride   // 16
        geom.vertexFormat = .float3
        geom.indexBuffer = ib
        geom.indexBufferOffset = 0
        geom.indexType = .uint32
        geom.triangleCount = rts.rtTriangleCount

        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [geom]
        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                            options: .storageModePrivate),
            let cmd = commandQueue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            Self.log.error("buildSoupAccel: AS build setup failed")
            return
        }
        storage.label = "Illuminatorama.rt.accel"
        enc.build(accelerationStructure: storage, descriptor: asDesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: one-time / topology-change AS build (not per-frame)
        Self.recordASBuildError(cmd, "buildSoupAccel.primAS")
        rts.rtAccel = storage
        rts.rtScratch = scratch
        // Curves live in a SEPARATE primitive AS (Metal forbids mixed-type
        // descriptors in one AS). Built with .refit so the per-frame sway kernel
        // can update the control points + refit in place.
        buildSoupCurveAccel()
    }

    /// Build the curve-only primitive AS (#60 item 7 incr. 2) for the active soup
    /// curve set. Separate from the triangle AS (Metal: one geometry type per
    /// primitive AS). `.refit` usage so `encodeCurveWindDisplaceAndRefit` can sway.
    func buildSoupCurveAccel() {
        rts.rtSoupCurveAccel = nil; rts.rtSoupRefitScratch = nil
        guard rts.rtSoupCurvesActive, let cgeom = makeSoupCurveGeom() else { return }
        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [cgeom]
        asDesc.usage = .refit
        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                            options: .storageModePrivate),
            let cmd = commandQueue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            Self.log.error("buildSoupCurveAccel: AS build setup failed; dropping curves.")
            rts.rtSoupCurvesActive = false; return
        }
        storage.label = "Illuminatorama.rt.curveAccel"
        enc.build(accelerationStructure: storage, descriptor: asDesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding(); cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: one-time / topology-change curve AS build
        Self.recordASBuildError(cmd, "buildSoupCurveAccel.curveAS")
        rts.rtSoupCurveAccel = storage
        // Scratch sized for BOTH refit and a full in-place rebuild — the per-frame
        // sway path REBUILDS the curve AS (cheap for the few-segment curve set),
        // confirmed to update the traced control points (the displace→rebuild→trace
        // +2m-offset probe moved the curves' shadows; whole-frame RMSE 0.0064).
        rts.rtSoupRefitScratch = device.makeBuffer(
            length: max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize, 16),
            options: .storageModePrivate)
    }

    /// Build the curve geometry descriptor for the active soup curve set from the
    /// pooled buffers. The control-point buffer is the AS-referenced (swayed) pool;
    /// the kernel writes this each frame and the AS refits to it.
    func makeSoupCurveGeom() -> MTLAccelerationStructureCurveGeometryDescriptor? {
        guard let cpts = rts.rtSoupCurvePoolPoints, let crad = rts.rtSoupCurveRadii,
              let cseg = rts.rtSoupCurveSegments, rts.rtSoupCurvePointCount > 0,
              rts.rtSoupCurveSegmentCount > 0 else { return nil }
        let g = MTLAccelerationStructureCurveGeometryDescriptor()
        g.controlPointBuffer = cpts
        g.controlPointCount = rts.rtSoupCurvePointCount
        g.controlPointStride = 12       // packed float3
        g.controlPointFormat = .float3
        g.radiusBuffer = crad
        g.radiusFormat = .float
        g.radiusStride = MemoryLayout<Float>.stride
        g.indexBuffer = cseg
        g.indexType = .uint32
        g.segmentCount = rts.rtSoupCurveSegmentCount
        g.segmentControlPointCount = 4
        g.curveType = .round            // exact 3D normals (branches)
        g.curveBasis = .catmullRom
        g.curveEndCaps = .sphere
        g.opaque = true
        return g
    }

    /// Adopt the process-wide curve registry for the SOUP path (#60 item 7 incr. 2).
    /// Called from `render()`'s non-TLAS branch before the RT passes. A registry
    /// version bump is a TOPOLOGY change here (the soup AS gains/replaces its curve
    /// geometry descriptor), so it rebuilds `rtAccel`. Requires BOTH curve-variant
    /// pipelines — if either fails to compile the set is dropped (a curve-holding
    /// AS traced by a triangle-contract intersector is UB, never shippable).
    func syncSoupCurveRegistry() {
        // TLAS-path renderers (extracted scenes) handle curves via syncCurveRegistry
        // in updateRTAccel; this is the soup/native path only.
        guard !buildRTFromExtractedScene else { return }
        let reg = IlluminatoramaCurveRegistry.shared
        guard reg.version != rts.rtSoupCurveSyncVersion else { return }
        rts.rtSoupCurveSyncVersion = reg.version
        // The soup path supports ONE curve set (single geometry descriptor). A
        // scene that needs many concatenates them into one set before registering.
        rts.rtSoupCurveSet = device.supportsRaytracing ? reg.sets.first : nil
        if rts.rtSoupCurveSet != nil {
            let cache = engine.pipelineCache
            var on = true
            if rts.rtSoupCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                rts.rtSoupCurvePipeline = cache.pipelineState(
                    name: "illumi_rt_lighting", device: device, constants: cv, variantKey: "soupCurves")
            }
            if scs.surfCacheSoupCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                scs.surfCacheSoupCurvePipeline = cache.pipelineState(
                    name: "illumi_surfcache_update", device: device, constants: cv, variantKey: "soupCurves")
            }
            if rts.rtSoupCurvePipeline == nil || scs.surfCacheSoupCurvePipeline == nil {
                Self.log.error("Soup curve RT: kRTCurvesEnabled pipeline variant failed; dropping curve set.")
                rts.rtSoupCurveSet = nil
            }
        }
        buildSoupCurvePool()
        rts.rtSoupCurvesActive = (rts.rtSoupCurveSet != nil)
            && rts.rtSoupCurvePoolPoints != nil && rts.rtSoupCurveRadii != nil
            && rts.rtSoupCurveSegments != nil && rts.rtSoupCurveSetDataBuffer != nil
        buildSoupAccel()   // rebuild the AS with / without the curve geometry descriptor
    }

    /// Pool the active soup curve set's control points (flat `[Float]` → read as
    /// `packed_float3`, stride 12), radii, segment indices (RELATIVE — one geometry
    /// descriptor), per-point wind attribute, and the single `RTCurveSetData`
    /// (identity transform — the soup set is world-space). Keeps a REST copy of the
    /// points so the per-frame wind kernel can re-displace from rest each frame.
    func buildSoupCurvePool() {
        rts.rtSoupCurveRestPoints = nil; rts.rtSoupCurvePoolPoints = nil; rts.rtSoupCurveRadii = nil
        rts.rtSoupCurveSegments = nil; rts.rtSoupCurveWindAttr = nil; rts.rtSoupCurveSetDataBuffer = nil
        rts.rtSoupCurvePointCount = 0; rts.rtSoupCurveSegmentCount = 0
        guard let set = rts.rtSoupCurveSet else { return }
        var pts: [Float] = []; pts.reserveCapacity(set.controlPoints.count * 3)
        for p in set.controlPoints { pts.append(p.x); pts.append(p.y); pts.append(p.z) }
        let data = [RTCurveSetData(
            m0: set.transform.columns.0, m1: set.transform.columns.1,
            m2: set.transform.columns.2, m3: set.transform.columns.3,
            albedoRoughness: SIMD4(set.material.albedo, set.material.roughness),
            emissionPad: SIMD4(set.material.emission, 0),
            meta: SIMD4<UInt32>(0, 0, 0, 0))]   // single set ⇒ segment base 0
        let ptLen = MemoryLayout<Float>.stride * pts.count
        guard ptLen > 0,
              let rest = device.makeBuffer(bytes: pts, length: ptLen, options: .storageModeShared),
              let pool = device.makeBuffer(bytes: pts, length: ptLen, options: .storageModeShared),
              let rb = device.makeBuffer(bytes: set.radii,
                  length: MemoryLayout<Float>.stride * set.radii.count, options: .storageModeShared),
              let sb = device.makeBuffer(bytes: set.segmentIndices,
                  length: MemoryLayout<UInt32>.stride * set.segmentIndices.count, options: .storageModeShared),
              let db = device.makeBuffer(bytes: data,
                  length: MemoryLayout<RTCurveSetData>.stride, options: .storageModeShared)
        else {
            Self.log.error("Soup curve RT: pool allocation failed; dropping curve set.")
            rts.rtSoupCurveSet = nil; return
        }
        rest.label = "Illuminatorama.rt.soupCurve.restPoints"
        pool.label = "Illuminatorama.rt.soupCurve.points"
        rb.label = "Illuminatorama.rt.soupCurve.radii"
        sb.label = "Illuminatorama.rt.soupCurve.segments"
        db.label = "Illuminatorama.rt.soupCurve.setData"
        rts.rtSoupCurveRestPoints = rest; rts.rtSoupCurvePoolPoints = pool
        rts.rtSoupCurveRadii = rb; rts.rtSoupCurveSegments = sb; rts.rtSoupCurveSetDataBuffer = db
        rts.rtSoupCurvePointCount = set.controlPoints.count
        rts.rtSoupCurveSegmentCount = set.segmentIndices.count
        // Per-point wind attribute (sway/phase/flutter) — only when the set opts in.
        if set.windAttr.count == set.controlPoints.count {
            rts.rtSoupCurveWindAttr = device.makeBuffer(bytes: set.windAttr,
                length: MemoryLayout<SIMD4<Float>>.stride * set.windAttr.count,
                options: .storageModeShared)
            rts.rtSoupCurveWindAttr?.label = "Illuminatorama.rt.soupCurve.windAttr"
        }
    }

    /// Per-frame curve sway (#60 item 7 incr. 2): re-displace the rest control
    /// points into the AS-referenced pool with the SAME `applyTreeWind` the vertex
    /// shader uses, then refit the soup AS. No-op for a rigid set (no windAttr) or
    /// under `VIZ_ILLUMI_NO_CURVE_REFIT=1`.
    func encodeCurveWindDisplaceAndRefit(_ cb: MTLCommandBuffer) {
        guard rts.rtSoupCurvesActive, !rts.noCurveRefit,
              let pipe = rts.curveWindDisplacePipeline,
              let rest = rts.rtSoupCurveRestPoints, let pool = rts.rtSoupCurvePoolPoints,
              let wind = rts.rtSoupCurveWindAttr, rts.rtSoupCurvePointCount > 0 else {
            if !rts.curveDbgPrinted, Self.frameEnv("VIZ_CURVE_DEBUG") == "1" {
                rts.curveDbgPrinted = true
                let m = "CURVE displace SKIPPED: active=\(rts.rtSoupCurvesActive) noRefit=\(rts.noCurveRefit) pipe=\(rts.curveWindDisplacePipeline != nil) rest=\(rts.rtSoupCurveRestPoints != nil) pool=\(rts.rtSoupCurvePoolPoints != nil) wind=\(rts.rtSoupCurveWindAttr != nil) pts=\(rts.rtSoupCurvePointCount)\n"
                FileHandle.standardError.write(m.data(using: .utf8)!)
            }
            return
        }
        if !rts.curveDbgPrinted, Self.frameEnv("VIZ_CURVE_DEBUG") == "1" {
            rts.curveDbgPrinted = true
            let m = "CURVE displace RAN: pts=\(rts.rtSoupCurvePointCount) time=\(time) windStrength=\(treeWindStrength) heading=\(treeWindHeading)\n"
            FileHandle.standardError.write(m.data(using: .utf8)!)
        }
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(rest, offset: 0, index: 0)
        enc.setBuffer(wind, offset: 0, index: 1)
        enc.setBuffer(pool, offset: 0, index: 2)
        var u = CurveWindUniforms(time: time, windStrength: treeWindStrength,
                                  windHeading: treeWindHeading,
                                  pointCount: UInt32(rts.rtSoupCurvePointCount))
        enc.setBytes(&u, length: MemoryLayout<CurveWindUniforms>.stride, index: 3)
        let w = min(pipe.threadExecutionWidth, rts.rtSoupCurvePointCount)
        enc.dispatchThreads(MTLSize(width: rts.rtSoupCurvePointCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
        enc.endEncoding()
        // Rebuild the CURVE-only AS in place from the displaced control points on
        // the SAME command buffer (Metal sequences the compute write → AS-build
        // read). A full rebuild is cheap for the few-segment curve set and
        // unambiguously re-reads the swayed pool (confirmed: the +2m-offset probe
        // moved the curves' shadows). gpu-ok: per-frame rebuild encoded on the
        // frame cb, no waitUntilCompleted.
        guard let accel = rts.rtSoupCurveAccel, let scratch = rts.rtSoupRefitScratch,
              let cgeom = makeSoupCurveGeom() else { return }
        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [cgeom]
        asDesc.usage = .refit
        guard let renc = cb.makeAccelerationStructureCommandEncoder() else { return }
        renc.build(accelerationStructure: accel, descriptor: asDesc,
                   scratchBuffer: scratch, scratchBufferOffset: 0)
        renc.endEncoding()
    }

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
                if let m = meshes[g.kind] { estTriangles += m.indexCount / 3 }
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
