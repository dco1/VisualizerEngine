import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {

    // Phase 4.39: SSR is now split into three passes.
    // 1) Gather  — ray march → write pre-weighted SSR delta to ssrRawTexture.
    // 2) Temporal — accumulate history with YCoCg variance clamp.
    // 3) Composite — hdrComposite = hdrTexture + denoisedSSR.
    func encodeSSRGather(_ cb: MTLCommandBuffer) {
        guard let enc = timedComputeEncoder(cb, "ssr.gather") else { return }
        enc.label = "Illuminatorama.ssr.gather"
        enc.setComputePipelineState(ssrGatherPipeline)
        enc.setTexture(hdrTexture,       index: 0)
        enc.setTexture(gbufferAlbedoMet, index: 1)
        enc.setTexture(gbufferNormalRgh, index: 2)
        enc.setTexture(depthTexture,     index: 3)
        enc.setTexture(ssrRawTexture,    index: 4)  // output: SSR delta
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrGatherPipeline, width: width, height: height)
        enc.endEncoding()
    }

    func encodeSSRTemporalPass(_ cb: MTLCommandBuffer) {
        guard ssrTemporalEnabled else { return }
        guard let enc = timedComputeEncoder(cb, "ssr.temporal") else { return }
        enc.label = "Illuminatorama.ssr.temporal"
        enc.setComputePipelineState(ssrTemporalPipeline)
        enc.setTexture(ssrRawTexture,             index: 0)  // current gather
        enc.setTexture(previousSSRHistoryTexture,  index: 1)  // history read
        enc.setTexture(velocityTexture,            index: 2)
        enc.setTexture(currentSSRHistoryTexture,   index: 3)  // history write
        enc.setTexture(ssrSampleCount,             index: 4)  // adaptive count (read_write)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrTemporalPipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// SSR-off fast path: seed `hdrCompositeTexture` from the lit frame with one
    /// blit (DMA copy — same size/format) in place of the three full-res SSR
    /// dispatches. Everything downstream keeps reading `hdrCompositeTexture`
    /// exactly as before.
    func encodeSSRBypassCopy(_ cb: MTLCommandBuffer) {
        guard let blit = cb.makeBlitCommandEncoder() else { return }
        blit.label = "Illuminatorama.ssr.bypass"
        blit.copy(from: hdrTexture, to: hdrCompositeTexture)
        blit.endEncoding()
    }

    func encodeSSRComposite(_ cb: MTLCommandBuffer) {
        guard let enc = timedComputeEncoder(cb, "ssr.composite") else { return }
        enc.label = "Illuminatorama.ssr.composite"
        enc.setComputePipelineState(ssrCompositePipeline)
        enc.setTexture(hdrTexture,        index: 0)  // base lighting
        enc.setTexture(ssrDenoisedTexture, index: 1)  // denoised SSR (or raw)
        enc.setTexture(hdrCompositeTexture, index: 2) // output
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrCompositePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// Issue #65 — true if a scene has opted into SSS this frame (matches the
    /// `effSSS` derivation in `uploadFrameUniforms`, env override included). Gates
    /// the whole blur/composite chain so non-SSS scenes encode NOTHING extra.
    private var sssEnabled: Bool {
        (Self.sssStrengthEnvOverride ?? sssStrength) > 0
    }

    /// Screen-space subsurface scattering (issue #65): separable blur of the
    /// diffuse-lit term the lighting pass peeled into `sssDiffuseTexture`, then a
    /// composite back into `hdrTexture` BEFORE SSR/RT/TAA read it. Three sub-passes:
    /// blur-H (sssDiffuse → sssBlurTmp) · blur-V (sssBlurTmp → sssBlurOut) ·
    /// composite (hdr += sssStrength·(sssBlurOut − sssDiffuse) on SSS pixels). No-op
    /// — and not encoded at all — unless a scene set `sssStrength > 0`.
    func encodeSSSPasses(_ cb: MTLCommandBuffer) {
        guard sssEnabled else { return }
        // Separable blur: one kernel, two passes, direction via setBytes.
        func blur(_ src: MTLTexture, _ dst: MTLTexture, dir: SIMD2<Float>, label: String) {
            guard let enc = timedComputeEncoder(cb, label) else { return }
            enc.label = "Illuminatorama.\(label)"
            enc.setComputePipelineState(sssBlurPipeline)
            enc.setTexture(src,          index: 0)
            enc.setTexture(depthTexture, index: 1)
            enc.setTexture(dst,          index: 2)
            enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
            var d = dir
            enc.setBytes(&d, length: MemoryLayout<SIMD2<Float>>.stride, index: 1)
            dispatch(enc, pipeline: sssBlurPipeline, width: width, height: height)
            enc.endEncoding()
        }
        blur(sssDiffuseTexture, sssBlurTmpTexture, dir: SIMD2(1, 0), label: "sss.blurH")
        blur(sssBlurTmpTexture, sssBlurOutTexture, dir: SIMD2(0, 1), label: "sss.blurV")

        guard let enc = timedComputeEncoder(cb, "sss.composite") else { return }
        enc.label = "Illuminatorama.sss.composite"
        enc.setComputePipelineState(sssCompositePipeline)
        enc.setTexture(hdrTexture,        index: 0)  // read_write: blurred diffuse blended in place
        enc.setTexture(sssDiffuseTexture, index: 1)  // sharp diffuse + mask
        enc.setTexture(sssBlurOutTexture, index: 2)  // blurred diffuse
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: sssCompositePipeline, width: width, height: height)
        enc.endEncoding()
    }
}
