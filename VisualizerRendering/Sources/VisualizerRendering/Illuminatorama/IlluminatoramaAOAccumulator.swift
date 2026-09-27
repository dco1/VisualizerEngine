import Metal
import simd

/// **Accumulated ray-traced AO — the still / settled-canvas AO pipeline (AO v2, Daydream DH-0887).**
///
/// Host opt-in through `IlluminatoramaRenderer.aoAccumulationEnabled`; with it off (the default)
/// nothing here is created, allocated or encoded, and every host renders byte-identically.
///
/// It exists because the per-frame RTAO path cannot converge a frozen camera well, for reasons
/// that are about ORDER, not budget: each frame's rays were drawn from a white-noise rotation,
/// blurred by a 3×3 bilateral, and only then averaged — by the lit-colour TAA, which clips its
/// history back toward each noisy frame. This object runs the standard progressive order instead
/// (see `IlluminatoramaAOAccumulate.metal`):
///
///     raw progressive OCCLUSION  →  ray-weighted running mean (fp32, no clamp)  →  edge filter
///     (easing from full to ½ as the mean fills)  →  ao = 1 − occlusion · strength
///
/// and hands the renderer's single AO selector (`aoSourceTexture`) the result.
///
/// **Reset rule.** The mean is valid only while nothing that shapes the AO FIELD changes. It
/// restarts on: a change of the UNJITTERED view-projection (the jittered one moves every frame on
/// purpose — accumulating across the jitter is the anti-aliasing), the AO size, the reach, the
/// traced scene's topology, the test scramble, a change of any instance's SHAPE (mesh, transform,
/// sway — materials and emission do not shape AO), being disabled, and `reset()`
/// (`resetTemporalHistory`, `resetAOAccumulation`). The frame on which the shape changed is not
/// accumulated either: the AO pass runs before this frame's acceleration-structure refit, so it
/// would trace the previous geometry. The grade's AO strength and the ray budget are NOT in the
/// key — occlusion is accumulated and the strength applied at the read, and each frame's weight is
/// its ray count. A frame whose AO fell back to the screen-space march (no TLAS yet) is not
/// accumulated and does not count.
@MainActor
final class IlluminatoramaAOAccumulator {

    /// Mirror of `RTAOv2Uniforms` (IlluminatoramaAOAccumulate.metal) — field for field, 128 bytes.
    struct RTAOv2Uniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD4<Float>
        var radius: Float
        var intensity: Float
        var rayTMin: Float
        var rayCount: UInt32
        var sampleBase: UInt32
        var scramble: UInt32
        var transportRayMask: UInt32
        var fullWidth: UInt32
        var fullHeight: UInt32
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }
    /// Mirror of `AOAccumulateUniforms` — 16 bytes.
    struct AccumulateUniforms {
        var weight: Float
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }
    /// Mirror of `AODenoiseUniforms` — 80 bytes.
    struct DenoiseUniforms {
        var invProjection: simd_float4x4
        var strength: Float
        var fullWidth: UInt32
        var fullHeight: UInt32
        var intensity: Float
    }

    /// Everything the accumulated field depends on. Equal keys ⇒ keep averaging.
    struct Key: Equatable {
        var unjitteredViewProjection: simd_float4x4
        var halfWidth: Int
        var halfHeight: Int
        var radius: Float
        var sceneTopology: Int
        var scramble: UInt32
    }

    let rtaoPipeline: MTLComputePipelineState?
    let accumulatePipeline: MTLComputePipelineState?
    let denoisePipeline: MTLComputePipelineState?

    /// Half-res fp32 running mean (lazily allocated on first use).
    private(set) var meanTexture: MTLTexture?
    /// Half-res r16Float denoised mean — the texture the lighting reads while this owns AO.
    private(set) var outputTexture: MTLTexture?
    /// Frames in the current mean (0 = nothing accumulated yet).
    private(set) var frames: Int = 0
    /// Rays per texel in the current mean — the next frame's first progressive sample index.
    private(set) var samples: Int = 0
    private var key: Key?

    var isAvailable: Bool { rtaoPipeline != nil && accumulatePipeline != nil && denoisePipeline != nil }

    init(device: MTLDevice, cache: SimPipelineCache) {
        // Optional pipelines, never `throw`: a compile failure here must not abort renderer init
        // for every host — it only leaves this opt-in path unavailable.
        rtaoPipeline = device.supportsRaytracing ? cache.pipelineState(name: "illumi_rtao_tlas_v2", device: device) : nil
        accumulatePipeline = cache.pipelineState(name: "illumi_ao_accumulate", device: device)
        denoisePipeline = cache.pipelineState(name: "illumi_ao_denoise", device: device)
    }

    func reset() {
        frames = 0
        samples = 0
        key = nil
    }

    /// Filter strength for a mean of `frames` frames: 1 on the first, easing to a floor of ½.
    /// Not to zero: on a smooth AO field the depth-edge 3×3's bias is 50–500× smaller than the
    /// residual noise it removes (measured on a crease model), so fading it out entirely would
    /// trade ~3× RMS for nothing visible.
    nonisolated static func denoiseStrength(frames: Int) -> Float { max(0.5, 4 / (4 + Float(max(0, frames - 1)))) }

    private func ensureTextures(device: MTLDevice, width: Int, height: Int) -> Bool {
        if let m = meanTexture, m.width == width, m.height == height, outputTexture != nil { return true }
        func make(_ format: MTLPixelFormat, _ label: String) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width,
                                                             height: height, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            let t = device.makeTexture(descriptor: d)
            t?.label = label
            return t
        }
        meanTexture = make(.r32Float, "Illuminatorama.ao.accum.mean")
        outputTexture = make(.r16Float, "Illuminatorama.ao.accum.out")
        reset()
        return meanTexture != nil && outputTexture != nil
    }

    /// Trace this frame's AO into `rawAO`, fold it into the mean, and denoise the mean into
    /// `outputTexture`. Returns false (and encodes nothing) when the path is unavailable, in
    /// which case the caller runs the per-frame path.
    ///
    /// `encoder` is the renderer's pass factory, so the passes show up in the per-pass timer.
    func encode(device: MTLDevice,
                encoder: (String) -> MTLComputeCommandEncoder?,
                dispatch: (MTLComputeCommandEncoder, MTLComputePipelineState, Int, Int) -> Void,
                depth: MTLTexture, rawAO: MTLTexture,
                tlas: MTLAccelerationStructure,
                key newKey: Key,
                shapeChanged: Bool,
                rays: Int,
                intensity: Float,
                uniforms: RTAOv2Uniforms,
                invProjection: simd_float4x4) -> Bool {
        guard let rtao = rtaoPipeline, let accumulate = accumulatePipeline,
              let denoise = denoisePipeline,
              ensureTextures(device: device, width: newKey.halfWidth, height: newKey.halfHeight),
              let mean = meanTexture, let out = outputTexture else { return false }
        // A shape change this frame: the traced geometry is last frame's, so restart the mean
        // and let this one frame take the per-frame path.
        if shapeChanged { reset(); return false }
        if key != newKey { reset(); key = newKey }

        let rayCount = max(1, rays)
        var u = uniforms
        u.rayCount = UInt32(rayCount)
        u.sampleBase = UInt32(truncatingIfNeeded: samples)
        if let enc = encoder("rtao") {
            enc.label = "Illuminatorama.rtao.v2"
            enc.setComputePipelineState(rtao)
            enc.setTexture(depth, index: 0)
            enc.setTexture(rawAO, index: 1)
            enc.setAccelerationStructure(tlas, bufferIndex: 0)
            enc.setBytes(&u, length: MemoryLayout<RTAOv2Uniforms>.stride, index: 1)
            dispatch(enc, rtao, newKey.halfWidth, newKey.halfHeight)
            enc.endEncoding()
        }
        var a = AccumulateUniforms(weight: Float(rayCount) / Float(samples + rayCount))
        if let enc = encoder("ao.accum") {
            enc.label = "Illuminatorama.ao.accum"
            enc.setComputePipelineState(accumulate)
            enc.setTexture(rawAO, index: 0)
            enc.setTexture(mean, index: 1)
            enc.setBytes(&a, length: MemoryLayout<AccumulateUniforms>.stride, index: 0)
            dispatch(enc, accumulate, newKey.halfWidth, newKey.halfHeight)
            enc.endEncoding()
        }
        frames += 1
        samples += rayCount
        var d = DenoiseUniforms(invProjection: invProjection,
                                strength: Self.denoiseStrength(frames: frames),
                                fullWidth: UInt32(depth.width), fullHeight: UInt32(depth.height),
                                intensity: max(0, min(1, intensity)))
        if let enc = encoder("ao.denoise") {
            enc.label = "Illuminatorama.ao.denoise"
            enc.setComputePipelineState(denoise)
            enc.setTexture(mean, index: 0)
            enc.setTexture(depth, index: 1)
            enc.setTexture(out, index: 2)
            enc.setBytes(&d, length: MemoryLayout<DenoiseUniforms>.stride, index: 0)
            dispatch(enc, denoise, newKey.halfWidth, newKey.halfHeight)
            enc.endEncoding()
        }
        return true
    }
}
