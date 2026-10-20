import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Diagnostic: append surface-cache atlas sizing to the sidecar file named by
    /// `VIZ_ILLUMI_SURFCACHE_STATS_PATH` (sandbox eats os_log). Used to quantify
    /// the P2/P3 atlas-efficiency wins before/after. No-op without the env var.
    nonisolated static func recordSurfCacheStats(_ line: String) {
        guard let p = frameEnv("VIZ_ILLUMI_SURFCACHE_STATS_PATH") else { return }
        guard let data = (line + "\n").data(using: .utf8) else { return }
        if let h = FileHandle(forWritingAtPath: p) { h.seekToEndOfFile(); h.write(data); try? h.close() }
        else { try? data.write(to: URL(fileURLWithPath: p)) }
    }

    /// Allocate (zeroed) on first use; nil only on allocation failure (counters then no-op).
    func surfCounterBuffer(_ existing: MTLBuffer?, slots: Int, label: String) -> MTLBuffer? {
        if let existing { return existing }
        let b = device.makeBuffer(length: MemoryLayout<UInt32>.stride * slots, options: .storageModeShared)
        b?.label = label
        if let b { memset(b.contents(), 0, b.length) }
        return b
    }

    /// Read + zero a counter buffer. Opt-in diagnostic lane only, so it may wait for the
    /// in-flight frame whose kernel is still adding (the feedback drain does the same).
    func drainSurfCounters(_ b: MTLBuffer, slots: Int) -> [UInt32] {
        awaitInFlightFrameBeforeCPUCacheWrite()
        let p = b.contents().bindMemory(to: UInt32.self, capacity: slots)
        let counts = (0..<slots).map { p[$0] }
        memset(b.contents(), 0, MemoryLayout<UInt32>.stride * slots)
        return counts
    }

    /// `hits:` sidecar line from the 8 hit slots (GI 0–3, reflection 4–7, each
    /// cached / fallback / noCard / reshade). nil when the frame made no surface hit.
    nonisolated static func surfCacheHitLine(_ c: [UInt32]) -> String? {
        guard c.count >= 8 else { return nil }
        let total = c.prefix(8).reduce(0) { $0 + Int($1) }
        guard total > 0 else { return nil }
        let cached = Int(c[0]) + Int(c[4])
        let rate = String(format: "%.1f", Double(cached) / Double(total) * 100)
        func group(_ o: Int) -> String {
            "cached=\(c[o]) fallback=\(c[o + 1]) noCard=\(c[o + 2]) reshade=\(c[o + 3])"
        }
        return "hits: cacheHitRate=\(rate)% cached=\(cached) of \(total) surface hits"
            + " gi[\(group(0))] refl[\(group(4))]"
    }

    /// `atlas:` sidecar line — texels temporally reused vs recomputed from scratch.
    nonisolated static func surfCacheAtlasLine(reused: UInt32, refreshed: UInt32) -> String? {
        let total = Int(reused) + Int(refreshed)
        guard total > 0 else { return nil }
        let rate = String(format: "%.1f", Double(reused) / Double(total) * 100)
        return "atlas: reuseRate=\(rate)% reused=\(reused) refreshed=\(refreshed) of \(total) texels"
    }

    /// Re-light the on-surface radiance atlas (ping-pong) before the RT pass.
    /// Reads PREVIOUS atlas for the indirect bounce (→ multi-bounce over frames);
    /// writes CURRENT. The RT pass then samples CURRENT at its hits.
    func encodeSurfaceCacheUpdate(_ cb: MTLCommandBuffer) {
        guard surfCacheActive,
              let cards = scs.surfCardBuffer, let tc = scs.surfTriCardBuffer,
              let ta = scs.surfTriUVaBuffer, let tcc = scs.surfTriUVcBuffer,
              let crect = scs.surfCardRectBuffer, let txcard = scs.surfTexelCardBuffer,
              let cdirty = scs.surfCardDirtyBuffer else { return }
        // §3 endpoint — trace the per-frame-refit TLAS when one is live (a moved
        // object's shadow/bounce-occlusion then tracks its motion); otherwise fall
        // back to the static-soup primitive AS (the soup-path scenes, no TLAS).
        let useTLAS = rts.rtTLASActive && scs.surfCacheTLASPipeline != nil
            && rts.rtTLAS != nil && rts.rtSoupTriBaseBuffer != nil
            && rts.rtSoupTriBaseCount == instances.count
        let pipeline: MTLComputePipelineState
        let accel: MTLAccelerationStructure
        if useTLAS {
            // Curve primitives (#60 item 7): a TLAS holding curve instances
            // must be traced with the curve-contract variant (curve hits then
            // occlude sun rays and terminate indirect rays — see the kernel).
            if rts.rtCurveInstanceCount > 0 {
                guard let cp = scs.surfCacheTLASCurvePipeline else { return }
                pipeline = cp
            } else {
                pipeline = scs.surfCacheTLASPipeline!
            }
            accel = rts.rtTLAS!
        } else {
            guard let soupAccel = rts.rtAccel else { return }
            // Curve primitives (#60 item 7 incr. 2): a soup AS holding a curve
            // geometry descriptor must be traced with the curve-contract variant
            // (curve hits occlude the sun ray + terminate indirect rays).
            if rts.rtSoupCurvesActive, let cp = scs.surfCacheSoupCurvePipeline {
                pipeline = cp
            } else if let soupPipe = scs.surfCachePipeline {
                pipeline = soupPipe
            } else { return }
            accel = soupAccel
        }
        // Honour per-card dirty flags (α=1 → re-light fresh at the re-framed pose)
        // for either dynamic source: rigid incremental invalidation OR Phase-C
        // deforming meshes (whose cards `applyIncrementalSurfaceCacheUpdate`
        // re-frames every frame). Both require the per-triangle CPU maps.
        let incremental = (surfaceCacheIncremental || scs.surfHasDeformingCards) && scs.surfIncrementalReady

        // DH-0622 — WARM-UP. A freshly baked (or freshly enabled) atlas holds no light, and every
        // bounce builds on the previous update's atlas, so at the normal blend a still's first frames
        // carry almost none of the multi-bounce light — measured cold on 4000 Sunset: +2.7 % after one
        // 32-frame export against a ~+22 % level-off. So the frame after a bake or an enable runs
        // `surfaceCacheWarmIterations` extra updates at α = 1 first: each carries the light one bounce
        // further, and the normal update then resumes smoothing from a filled atlas.
        let warm = scs.surfCacheWarmPending
        scs.surfCacheWarmPending = 0

        // DH-0653 — atlas reuse counter: report the previous frame's, arm this frame's final update.
        var atlasStats: MTLBuffer? = nil
        if Self.surfCacheStatsRequested {
            scs.surfAtlasStatsBuffer = surfCounterBuffer(scs.surfAtlasStatsBuffer, slots: 2,
                                                     label: "Illuminatorama.surfcache.atlasStats")
            if let sb = scs.surfAtlasStatsBuffer {
                let c = drainSurfCounters(sb, slots: 2)
                if let line = Self.surfCacheAtlasLine(reused: c[0], refreshed: c[1]) {
                    Self.recordSurfCacheStats(line)
                }
                atlasStats = sb
            }
        }

        for pass in 0...warm {
            let warming = pass < warm
            // Ping-pong swap at the START (mirrors DDGI): Current is what we write
            // AND what the RT pass reads after we return; Previous is the bounce
            // source. First frame after (re)alloc both are empty → indirect = 0.
            scs.surfUseAtlasA.toggle()
            guard let curAtlas = surfCacheCurrentAtlas,
                  let prevAtlas = surfCachePreviousAtlas,
                  let curIndirect = surfCacheCurrentIndirectAtlas,
                  let prevIndirect = surfCachePreviousIndirectAtlas else { return }

            scs.surfFrameSeed &+= 1
            var u = SurfCacheUniforms(
                sunDir: SIMD4(simd_normalize(rtSunDirection), 0),
                sunColor: SIMD4(rtSunColor, 0),
                skyAmbient: SIMD4(ambientColor, 0),
                atlasW: UInt32(scs.surfAtlasW), atlasH: UInt32(scs.surfAtlasH),
                tileSize: UInt32(scs.surfActiveTileSize), tilesPerRow: UInt32(scs.surfTilesPerRow),
                cardCount: UInt32(scs.surfCardCount), triangleCount: UInt32(rts.rtTriangleCount),
                indirectRays: UInt32(max(1, min(16, surfCacheIndirectRays))),
                frameSeed: scs.surfFrameSeed,
                alpha: warming ? 1.0 : max(0.02, min(1.0, surfCacheAlpha)), rayTMin: 0.004, maxDist: 60.0,
                incrementalEnabled: incremental ? 1 : 0)
            if !warming, atlasStats != nil { u.statsEnabled = 1 }
            u.backFaceGuard = surfaceCacheBackFaceGuard ? 1 : 0
            // DH-0872 — same coefficient the tonemap pass uses, applied to this pass's own
            // raw sky-miss sample (see the kernel's miss branch).
            u.scotopicDesaturation = max(0, scotopicDesaturation)
            u.cardlessReshade = surfaceCacheCardlessReshade ? 1 : 0
            memcpy(scs.surfCacheUniformBuffer.contents(), &u, MemoryLayout<SurfCacheUniforms>.stride)

            // Warm passes go untimed: the per-pass GPU timer holds 48 passes a frame.
            guard let enc = warming ? cb.makeComputeCommandEncoder() : timedComputeEncoder(cb, "surfcacheUpdate")
            else { return }
            enc.label = warming ? "Illuminatorama.surfcache.warm" : "Illuminatorama.surfcache"
            enc.setComputePipelineState(pipeline)
            enc.setTexture(curAtlas, index: 0)
            enc.setTexture(prevAtlas, index: 1)   // read access
            enc.setTexture(prevAtlas, index: 2)   // sample access
            enc.setTexture(equirectSky ?? ibls.dummySkyTexture, index: 3)
            enc.setTexture(curIndirect, index: 4)    // DH-0622 — write: the indirect-only EMA
            enc.setTexture(prevIndirect, index: 5)   // read
            enc.setAccelerationStructure(accel, bufferIndex: 0)
            enc.setBuffer(cards, offset: 0, index: 1)
            enc.setBuffer(tc, offset: 0, index: 2)
            enc.setBuffer(ta, offset: 0, index: 3)
            enc.setBuffer(tcc, offset: 0, index: 4)
            // Bytes per pass, not the shared uniform buffer: a buffer is read when the command buffer
            // EXECUTES, so every pass encoded into this frame would see the last pass's α and seed.
            enc.setBytes(&u, length: MemoryLayout<SurfCacheUniforms>.stride, index: 5)
            enc.setBuffer(crect, offset: 0, index: 6)
            enc.setBuffer(txcard, offset: 0, index: 7)
            enc.setBuffer(cdirty, offset: 0, index: 8)
            // DH-0653 counters (buffer 10, both variants). Dummy = the dirty buffer, which
            // the kernel never adds to while `statsEnabled` is 0.
            enc.setBuffer(atlasStats ?? cdirty, offset: 0, index: 10)
            if useTLAS {
                // TLAS hit → global soup tri via soupTriBase[instance_id]+primitive_id.
                // The TLAS references the BLASes, which reference the mesh vertex/index
                // buffers (and the curve BLASes the pooled curve buffers) — all must
                // be resident for the intersector.
                enc.setBuffer(rts.rtSoupTriBaseBuffer, offset: 0, index: 9)
                // DH-1314 — a cardless hit is re-shaded from the RT instance rows + object normals.
                enc.setBuffer(rts.rtInstanceDataBuffer ?? cdirty, offset: 0, index: 11)
                enc.setBuffer(rts.rtObjNormalBuffer ?? cdirty, offset: 0, index: 12)
                for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
                for blas in rts.rtCurveBLASList { enc.useResource(blas, usage: .read) }
                for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
            } else if let vb = rts.rtVertexBuffer, let ib = rts.rtIndexBuffer {
                // Curve-only primitive AS at buffer(9) — the SOUP kernel's slot (the
                // TLAS variant uses 9 for soupTriBase; different pipeline). Dummy =
                // the triangle `accel` for the base (curve-free) variant.
                enc.setAccelerationStructure((rts.rtSoupCurvesActive ? rts.rtSoupCurveAccel : nil) ?? accel, bufferIndex: 9)
                enc.useResource(vb, usage: .read)
                enc.useResource(ib, usage: .read)
                // Curve AS + pool buffers the soup curve trace references (#60 item 7 incr. 2).
                if rts.rtSoupCurvesActive {
                    if let ca = rts.rtSoupCurveAccel { enc.useResource(ca, usage: .read) }
                    if let p = rts.rtSoupCurvePoolPoints { enc.useResource(p, usage: .read) }
                    if let r = rts.rtSoupCurveRadii { enc.useResource(r, usage: .read) }
                    if let s = rts.rtSoupCurveSegments { enc.useResource(s, usage: .read) }
                }
            }
            dispatch(enc, pipeline: pipeline, width: scs.surfAtlasW, height: scs.surfAtlasH)
            enc.endEncoding()
        }
    }

    // Phase 5 / B1 — cache-domain à-trous denoiser. Mirror of the Metal
    // `SurfAtrousUniforms` (stride 32).
    private struct SurfAtrousUniforms {
        var atlasW: UInt32; var atlasH: UInt32; var stepSize: UInt32; var radius: UInt32
        var phi: Float; var _ap0: Float = 0; var _ap1: Float = 0; var _ap2: Float = 0
    }

    /// Phase 5 / B1 — spatially denoise the just-updated radiance atlas into
    /// `surfAtlasDenoised`, which the GI/reflection consumers then read. Encoded
    /// AFTER `encodeSurfaceCacheUpdate` (current atlas written) and BEFORE the RT
    /// lighting pass (same command buffer ⇒ compute passes serialize). No-op when
    /// the denoiser is off, the pipeline is unavailable, or the textures aren't
    /// allocated — consumers fall back to the raw atlas via `surfConsumerAtlas`.
    func encodeSurfaceCacheDenoise(_ cb: MTLCommandBuffer) {
        guard surfaceCacheDenoise,
              let pipeline = scs.surfAtrousPipeline,
              let src = surfCacheCurrentIndirectAtlas,
              let dst = scs.surfAtlasDenoised,
              let txcard = scs.surfTexelCardBuffer,
              scs.surfCardCount > 0 else { return }
        guard let enc = timedComputeEncoder(cb, "surfcacheDenoise") else { return }
        enc.label = "Illuminatorama.surfcache.atrous"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(dst, index: 0)   // write
        enc.setTexture(src, index: 1)   // read (current atlas)
        enc.setBuffer(txcard, offset: 0, index: 0)
        var u = SurfAtrousUniforms(
            atlasW: UInt32(scs.surfAtlasW), atlasH: UInt32(scs.surfAtlasH),
            stepSize: 1, radius: 2, phi: 4.0)
        enc.setBytes(&u, length: MemoryLayout<SurfAtrousUniforms>.stride, index: 1)
        dispatch(enc, pipeline: pipeline, width: scs.surfAtlasW, height: scs.surfAtlasH)
        enc.endEncoding()
    }
}
