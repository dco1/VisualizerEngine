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
        enc.setComputePipelineState(ssrs.ssrGatherPipeline)
        enc.setTexture(gbs.hdrTexture,       index: 0)
        enc.setTexture(gbs.gbufferAlbedoMet, index: 1)
        enc.setTexture(gbs.gbufferNormalRgh, index: 2)
        enc.setTexture(gbs.depthTexture,     index: 3)
        enc.setTexture(ssrs.ssrRawTexture,    index: 4)  // output: SSR delta
        enc.setTexture(gbs.gbufferMaterialLayer, index: 5)          // DH-1300 — the material-layer slot
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 1)   // …and the instances it indexes
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrs.ssrGatherPipeline, width: width, height: height)
        enc.endEncoding()
    }

    func encodeSSRTemporalPass(_ cb: MTLCommandBuffer) {
        guard ssrTemporalEnabled else { return }
        guard let enc = timedComputeEncoder(cb, "ssr.temporal") else { return }
        enc.label = "Illuminatorama.ssr.temporal"
        enc.setComputePipelineState(ssrs.ssrTemporalPipeline)
        enc.setTexture(ssrs.ssrRawTexture,             index: 0)  // current gather
        enc.setTexture(previousSSRHistoryTexture,  index: 1)  // history read
        enc.setTexture(pfx.velocityTexture,            index: 2)
        enc.setTexture(currentSSRHistoryTexture,   index: 3)  // history write
        enc.setTexture(ssrs.ssrSampleCount,             index: 4)  // adaptive count (read_write)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrs.ssrTemporalPipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// SSR-off fast path: seed `hdrCompositeTexture` from the lit frame with one
    /// blit (DMA copy — same size/format) in place of the three full-res SSR
    /// dispatches. Everything downstream keeps reading `hdrCompositeTexture`
    /// exactly as before.
    func encodeSSRBypassCopy(_ cb: MTLCommandBuffer) {
        guard let blit = cb.makeBlitCommandEncoder() else { return }
        blit.label = "Illuminatorama.ssr.bypass"
        blit.copy(from: gbs.hdrTexture, to: pfx.hdrCompositeTexture)
        blit.endEncoding()
    }

    func encodeSSRComposite(_ cb: MTLCommandBuffer) {
        guard let enc = timedComputeEncoder(cb, "ssr.composite") else { return }
        enc.label = "Illuminatorama.ssr.composite"
        enc.setComputePipelineState(ssrs.ssrCompositePipeline)
        enc.setTexture(gbs.hdrTexture,        index: 0)  // base lighting
        enc.setTexture(ssrDenoisedTexture, index: 1)  // denoised SSR (or raw)
        enc.setTexture(pfx.hdrCompositeTexture, index: 2) // output
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrs.ssrCompositePipeline, width: width, height: height)
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
            enc.setComputePipelineState(ssrs.sssBlurPipeline)
            enc.setTexture(src,          index: 0)
            enc.setTexture(gbs.depthTexture, index: 1)
            enc.setTexture(dst,          index: 2)
            enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
            var d = dir
            enc.setBytes(&d, length: MemoryLayout<SIMD2<Float>>.stride, index: 1)
            dispatch(enc, pipeline: ssrs.sssBlurPipeline, width: width, height: height)
            enc.endEncoding()
        }
        blur(ssrs.sssDiffuseTexture, ssrs.sssBlurTmpTexture, dir: SIMD2(1, 0), label: "sss.blurH")
        blur(ssrs.sssBlurTmpTexture, ssrs.sssBlurOutTexture, dir: SIMD2(0, 1), label: "sss.blurV")

        guard let enc = timedComputeEncoder(cb, "sss.composite") else { return }
        enc.label = "Illuminatorama.sss.composite"
        enc.setComputePipelineState(ssrs.sssCompositePipeline)
        enc.setTexture(gbs.hdrTexture,        index: 0)  // read_write: blurred diffuse blended in place
        enc.setTexture(ssrs.sssDiffuseTexture, index: 1)  // sharp diffuse + mask
        enc.setTexture(ssrs.sssBlurOutTexture, index: 2)  // blurred diffuse
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: ssrs.sssCompositePipeline, width: width, height: height)
        enc.endEncoding()
    }
}

// MARK: - SSRSSSState

/// Screen-space reflections and screen-space subsurface scattering.
/// Owned by the renderer as `ssrs`; every member keeps the name it had on the renderer.
@MainActor
final class SSRSSSState {
    // SSR spatiotemporal: raw gather → temporal → composite into hdrComposite.
    var ssrRawTexture: MTLTexture!  // full-res, rgba16Float
    var ssrHistoryA: MTLTexture!  // full-res, rgba16Float ping-pong
    var ssrHistoryB: MTLTexture!
    var ssrHistoryToggle: Bool = false
    /// Tracks `ssrIntensity > 0` transitions so a re-enable re-primes the SSR
    /// history (the whole SSR chain is skipped while intensity is 0).
    var previousSSRIntensityActive: Bool = false
    var ssrNeedsFirstFrame: Bool = true
    var previousSsrEnabled: Bool = false

    // Issue #65 — screen-space SSS buffers (full-res, rgba16Float). The lighting
    // pass writes the diffuse-lit term of SSS-flagged pixels into `sssDiffuse`;
    // the separable blur ping-pongs sssDiffuse → sssBlurTmp (H) → sssBlurOut (V);
    // the composite reads sssDiffuse (sharp) + sssBlurOut (blurred). Idle unless a
    // scene sets `sssStrength > 0` (the blur/composite passes aren't even encoded).
    var sssDiffuseTexture: MTLTexture!
    var sssBlurTmpTexture: MTLTexture!
    var sssBlurOutTexture: MTLTexture!

    var ssrSampleCount: MTLTexture!  // full-res

    var ssrGatherPipeline: MTLComputePipelineState!  // was ssrPipeline; now writes ssrRaw

    var ssrTemporalPipeline: MTLComputePipelineState!
    var ssrCompositePipeline: MTLComputePipelineState!
    // Issue #65 — screen-space SSS. One blur kernel run twice (H then V, direction
    // passed via setBytes) + a composite kernel.
    var sssBlurPipeline: MTLComputePipelineState!
    var sssCompositePipeline: MTLComputePipelineState!
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    // Issue #65 — headless SSS override. SSS is otherwise driven only by the panel.
    // `VIZ_ILLUMI_SSS` (a float strength, e.g. `1.0`) forces `sssStrength` for every
    // renderer in the process so the light-bleed can be render-verified A/B; the
    // optional `VIZ_ILLUMI_SSS_FORCEALL=1` companion treats EVERY opaque pixel as
    // SSS-flagged (so any scene's props can be verified without re-tagging a mesh).
    // `nil` when unset.
    nonisolated static let sssStrengthEnvOverride: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SSS"] else { return nil }
        return Float(v)
    }()

    nonisolated static let sssForceAllEnv: Bool =
        (ProcessInfo.processInfo.environment["VIZ_ILLUMI_SSS_FORCEALL"] == "1")

    // Optional headless radius override (mm), so a verify run can crank the
    // diffusion reach without re-authoring panel state. `nil` when unset.
    nonisolated static let sssRadiusEnvOverride: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SSS_RADIUS"] else { return nil }
        return Float(v)
    }()
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// SSR gather/temporal/composite and SSS blur/composite compute pipelines.
    @MainActor static func buildSSRSSSPipelines(into s: SSRSSSState, cache: SimPipelineCache, device: MTLDevice) throws {
        guard let ssrGatherPipelineBuilt = cache.pipelineState(name: "illumi_ssr_gather", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssr_gather")
        }
        s.ssrGatherPipeline = ssrGatherPipelineBuilt
        guard let ssrTemporalPipelineBuilt = cache.pipelineState(name: "illumi_ssr_temporal", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssr_temporal")
        }
        s.ssrTemporalPipeline = ssrTemporalPipelineBuilt
        guard let ssrCompositePipelineBuilt = cache.pipelineState(name: "illumi_ssr_composite", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_ssr_composite")
        }
        s.ssrCompositePipeline = ssrCompositePipelineBuilt
        guard let sssBlurPipelineBuilt = cache.pipelineState(name: "illumi_sss_blur", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_sss_blur")
        }
        s.sssBlurPipeline = sssBlurPipelineBuilt
        guard let sssCompositePipelineBuilt = cache.pipelineState(name: "illumi_sss_composite", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_sss_composite")
        }
        s.sssCompositePipeline = sssCompositePipelineBuilt
    }
}
