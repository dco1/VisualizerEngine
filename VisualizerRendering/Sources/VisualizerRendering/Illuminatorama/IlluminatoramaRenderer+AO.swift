import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {

    /// Make (once) and clear the 1×1 neutral-white AO texture. A clear-only
    /// render pass is the cheapest portable way to put a known value into a
    /// `.private` texture — the same idiom `encodeBlankSkyPass` already uses —
    /// and it costs one encoder, one time, only on a frame where SSAO is off.
    /// Returns false if the texture could not be allocated, in which case the
    /// caller must fall back to actually running the SSAO passes.
    @discardableResult
    private func ensureNeutralAOTexture(_ cb: MTLCommandBuffer) -> Bool {
        if aoNeutralTexture != nil { return true }
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        d.pixelFormat = .r16Float          // matches the AO chain's format
        d.width = 1
        d.height = 1
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let tex = device.makeTexture(descriptor: d) else {
            Self.log.error("Illuminatorama: neutral AO texture allocation failed — SSAO skip disabled")
            return false
        }
        tex.label = "Illuminatorama.ao.neutral"
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = tex
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return false }
        enc.label = "Illuminatorama.ao.neutralClear"
        enc.endEncoding()
        aoNeutralTexture = tex
        return true
    }

    func encodeSSAOPass(_ cb: MTLCommandBuffer) {
        // Skip the whole SSAO chain when it can only produce 1.0. The lighting
        // pass reads `aoNeutralTexture` instead (see `aoSourceTexture`).
        aoAccumulatorOwnsAO = false
        if !ssaoActive, ensureNeutralAOTexture(cb) { return }
        let halfW = max(1, width / 2)
        let halfH = max(1, height / 2)
        if encodeAccumulatedRTAO(cb, halfW: halfW, halfH: halfH) { return }
        // RTAO (DH-0528): with a live TLAS and the host opted in, trace the near field against
        // the real geometry into the SAME raw AO texture, so the bilateral + temporal chain and
        // the lighting kernel's read are untouched. A frame without a TLAS (frame 1, a tripped
        // guard) falls back to the screen-space march below, and says so.
        if rtaoActive, let tlas = rtTLAS, let pipeline = rtaoPipeline,
           let enc = timedComputeEncoder(cb, "rtao") {
            enc.label = "Illuminatorama.rtao"
            enc.setComputePipelineState(pipeline)
            enc.setTexture(depthTexture, index: 0)
            enc.setTexture(gbufferNormalRgh, index: 1)
            enc.setTexture(aoTexture, index: 2)
            enc.setAccelerationStructure(tlas, bufferIndex: 0)
            let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
            var u = RTAOUniforms(invViewProjection: fu.invViewProjection,
                                 radius: max(0.02, rtaoRadius),
                                 intensity: max(0, min(1, rtaoIntensity)),
                                 // DH-0888 — the budget the caller ASKED for. This used to be
                                 // silently `min(32, …)`: a public property documented as the
                                 // ray count, quietly capped three levels down, with no
                                 // assertion and no log. Production never exceeds it
                                 // (`settleRTAORays` 4, `photoRTAORays` 8) so no shipped frame
                                 // changes, but it capped a diagnostic sweep and made a 512-ray
                                 // arm read as converged when it had traced 32. A ceiling stays
                                 // — a runaway value would hang the GPU — but at a number no
                                 // legitimate caller reaches, and the property means what it says
                                 // below it.
                                 rayCount: UInt32(max(1, min(4096, rtaoRays))),
                                 // Same seed contract as every other traced term: walk only
                                 // while an accumulator can average it, else freeze.
                                 frameSeed: rtaoFrameSeed,
                                 rayTMin: 0.004,
                                 transportRayMask: 0x01 | 0x04,
                                 fullWidth: UInt32(width), fullHeight: UInt32(height))
            enc.setBytes(&u, length: MemoryLayout<RTAOUniforms>.stride, index: 1)
            dispatch(enc, pipeline: pipeline, width: halfW, height: halfH)
            enc.endEncoding()
            rtaoFrameSeed &+= 1
            rtaoDidRunLastFrame = true
            return
        }
        rtaoDidRunLastFrame = false
        guard let enc = timedComputeEncoder(cb, "ssao") else { return }
        enc.label = "Illuminatorama.ssao"
        enc.setComputePipelineState(ssaoPipeline)
        enc.setTexture(depthTexture, index: 0)
        enc.setTexture(gbufferNormalRgh, index: 1)
        enc.setTexture(aoTexture, index: 2)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssaoPipeline, width: halfW, height: halfH)
        enc.endEncoding()
    }

    /// AO v2 (Daydream DH-0887): the accumulated traced AO, when the host opted in and a TLAS is
    /// live. Returns false — encoding nothing — otherwise, and the per-frame path runs as before.
    /// A frame that falls back to GTAO (no TLAS yet) is not accumulated and does not count.
    private func encodeAccumulatedRTAO(_ cb: MTLCommandBuffer, halfW: Int, halfH: Int) -> Bool {
        guard aoAccumulationEnabled, rtaoActive, let tlas = rtTLAS else { return false }
        if aoAccumulator == nil {
            aoAccumulator = IlluminatoramaAOAccumulator(device: device, cache: engine.pipelineCache)
        }
        guard let acc = aoAccumulator, acc.isAvailable else { return false }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        let rays = max(1, min(4096, rtaoRays))
        let radius = max(0.02, rtaoRadius)
        let intensity = max(0, min(1, rtaoIntensity))
        let key = IlluminatoramaAOAccumulator.Key(
            unjitteredViewProjection: camera.projectionMatrix * camera.viewMatrix,
            halfWidth: halfW, halfHeight: halfH, radius: radius,
            sceneTopology: rtTLASTopologyHash, scramble: aoAccumulationScramble)
        let u = IlluminatoramaAOAccumulator.RTAOv2Uniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: SIMD4(fu.cameraWorldPos, 1),
            radius: radius, intensity: intensity, rayTMin: 0.001,
            rayCount: UInt32(rays), sampleBase: 0, scramble: aoAccumulationScramble,
            transportRayMask: 0x01 | 0x04,
            fullWidth: UInt32(width), fullHeight: UInt32(height),
            surfaceSnapULPs: effectiveRTSurfaceSnapULPs)
        let encoded = acc.encode(device: device,
                                 encoder: { [unowned self] label in self.timedComputeEncoder(cb, label) },
                                 dispatch: { [unowned self] enc, p, w, h in self.dispatch(enc, pipeline: p, width: w, height: h) },
                                 depth: depthTexture, rawAO: aoTexture, tlas: tlas,
                                 key: key, shapeChanged: instanceShapeStableFrames == 0,
                                 rays: rays, intensity: intensity,
                                 uniforms: u, invProjection: fu.invProjection,
                                 perFrame: aoStillPerFrame && taaStillAccumulation)
        guard encoded else { return false }
        rtaoDidRunLastFrame = true
        aoAccumulatorOwnsAO = true
        return true
    }

    func encodeSSAOSpatialFilter(_ cb: MTLCommandBuffer) {
        guard ssaoSpatialRuns, ssaoChainEncodes, !aoAccumulatorOwnsAO else { return }
        let halfW = max(1, width / 2)
        let halfH = max(1, height / 2)
        guard let enc = timedComputeEncoder(cb, "ssao.spatial") else { return }
        enc.label = "Illuminatorama.ssao.spatial"
        enc.setComputePipelineState(ssaoSpatialPipeline)
        enc.setTexture(aoTexture,        index: 0)   // raw AO input
        enc.setTexture(depthTexture,     index: 1)
        enc.setTexture(gbufferNormalRgh, index: 2)
        enc.setTexture(aoFilteredTexture, index: 3)  // output
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssaoSpatialPipeline, width: halfW, height: halfH)
        enc.endEncoding()
    }

    func encodeSSAOTemporalPass(_ cb: MTLCommandBuffer) {
        guard ssaoTemporalEnabled, ssaoChainEncodes, !aoAccumulatorOwnsAO else { return }
        let halfW = max(1, width / 2)
        let halfH = max(1, height / 2)
        guard let enc = timedComputeEncoder(cb, "ssao.temporal") else { return }
        enc.label = "Illuminatorama.ssao.temporal"
        enc.setComputePipelineState(ssaoTemporalPipeline)
        enc.setTexture(aoFilteredTexture,        index: 0)  // filtered current
        enc.setTexture(previousAOHistoryTexture,  index: 1)  // history read
        enc.setTexture(velocityTexture,           index: 2)
        enc.setTexture(currentAOHistoryTexture,   index: 3)  // history write
        enc.setTexture(aoSampleCount,             index: 4)  // adaptive count (read_write)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssaoTemporalPipeline, width: halfW, height: halfH)
        enc.endEncoding()
    }
}
