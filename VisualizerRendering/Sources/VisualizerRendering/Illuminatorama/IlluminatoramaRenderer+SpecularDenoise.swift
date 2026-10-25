import Foundation
import Metal
import simd
import VisualizerCore

// MARK: - DH-0539 — the RT reflection (specular) denoiser
//
// The diffuse RT term has a denoise chain; the glossy reflection term used to be added RAW into the HDR
// (`illumi_rt_lighting_tlas`), so only TAA ever cleaned it and the host capped the roughness cutoff at
// 0.25. With `rtSpecularDenoise` on, the TLAS kernel writes the reflection (minus the sky it replaces) to
// `spec.raw` and this chain — hit-distance-aware temporal reprojection, a roughness-scaled variance-guided
// à-trous, a composite — puts it back. Shaders: IlluminatoramaSpecDenoise.metal. Default OFF
// (byte-identical); TLAS path only (the soup kernel keeps the raw add).

/// The chain's textures + knobs. Allocated lazily at the render size, only while the lever is on.
@MainActor
final class SpecDenoiseState {
    var enabled = false
    var wasEnabledLastFrame = false
    var levels = 4
    var mirrorRoughness: Float = 0.06
    var fullRoughness: Float = 0.45
    var alphaRough: Float = 0.1
    var clampSigma: Float = 1.5
    var sigmaL: Float = 4
    var sigmaZ: Float = 1
    var sigmaN: Float = 64
    var stillLumFloor: Float = 100

    var width = 0, height = 0
    var raw: MTLTexture!
    var fres: MTLTexture!                              // Fresnel × strength the signal was divided by
    var colorA: MTLTexture!, colorB: MTLTexture!      // temporal colour ping-pong (also à-trous level 0 input)
    var auxA: MTLTexture!, auxB: MTLTexture!          // (m1, m2, count, viewZ) ping-pong, rgba32Float
    var varT: MTLTexture!, varA: MTLTexture!, varB: MTLTexture!
    var atrousA: MTLTexture!, atrousB: MTLTexture!
    var standIn: MTLTexture!                           // 1×1 stand-in for the history a still does not keep
    var lastStill = false
    var toggle = false
    var needsFirstFrame = true
    var writtenThisFrame = false

    var temporalPipeline: MTLComputePipelineState?
    var atrousPipeline: MTLComputePipelineState?
    var compositePipeline: MTLComputePipelineState?

    struct Uniforms {
        var width: UInt32; var height: UInt32; var flags: UInt32; var stepSize: UInt32
        var sigmaL: Float; var sigmaZ: Float; var sigmaN: Float; var clampSigma: Float
        var mirrorRough: Float; var fullRough: Float; var alphaRough: Float; var maxLevels: UInt32
        var lumFloor: Float; var pad0: Float = 0, pad1: Float = 0, pad2: Float = 0
    }
}

extension IlluminatoramaRenderer {
    /// Route the RT glossy reflection through the specular denoise chain (TLAS path). Default false.
    public var rtSpecularDenoise: Bool {
        get { rts.spec.enabled } set { rts.spec.enabled = newValue }
    }
    /// à-trous levels run for a fully rough pixel (a mirror runs none). 1…5.
    public var rtSpecularDenoiseLevels: Int {
        get { rts.spec.levels } set { rts.spec.levels = max(1, min(5, newValue)) }
    }
    /// At or below this roughness a lobe is a mirror: no temporal history, no spatial filter.
    public var rtSpecularMirrorRoughness: Float {
        get { rts.spec.mirrorRoughness } set { rts.spec.mirrorRoughness = max(0, newValue) }
    }
    /// Luminance-stop bandwidth floor in a still (relative to the pixel's own level). The default (100) is
    /// "no luminance stop": measured against a converged reference, a stop that keys on a NOISY neighbour
    /// darkens the mean 0.4–0.7 % (it drops exactly the bright samples the mean needs) and gains nothing —
    /// depth / normal / roughness stops hold the geometry, and a still runs ONE level.
    public var rtSpecularStillLumFloor: Float {
        get { rts.spec.stillLumFloor } set { rts.spec.stillLumFloor = max(0.01, newValue) }
    }
    /// History weight of the CURRENT frame for a fully rough lobe (smaller ⇒ longer history).
    public var rtSpecularRoughAlpha: Float {
        get { rts.spec.alphaRough } set { rts.spec.alphaRough = max(0.01, min(1, newValue)) }
    }

    /// True when the TLAS pass should write `spec.raw` this frame; (re)allocates the targets.
    func prepareSpecularDenoise() -> Bool {
        let s = rts.spec
        guard s.enabled, rtReflectionsEnabled, rts.rtTLASActive, rtEnabled, rtOpaqueLightingEnabled else {
            s.writtenThisFrame = false
            return false
        }
        if s.width != width || s.height != height || s.raw == nil {
            func make(_ label: String, _ fmt: MTLPixelFormat) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: width, height: height, mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
                let t = device.makeTexture(descriptor: d); t?.label = "Illuminatorama.spec." + label
                return t
            }
            guard let raw = make("raw", .rgba16Float), let fr = make("fres", .rgba16Float), let ca = make("colorA", .rgba16Float),
                  let vt = make("varT", .r16Float), let va = make("varA", .r16Float), let vb = make("varB", .r16Float),
                  let aa = make("atrousA", .rgba16Float), let ab = make("atrousB", .rgba16Float) else { return false }
            s.raw = raw; s.fres = fr; s.colorA = ca; s.colorB = nil; s.auxA = nil; s.auxB = nil
            s.varT = vt; s.varA = va; s.varB = vb; s.atrousA = aa; s.atrousB = ab
            s.width = width; s.height = height; s.needsFirstFrame = true
        }
        // The temporal history (a second colour buffer + the moments) exists only where there IS a history: a
        // frozen still averages its frames in the colour resolve and keeps none, which at the export's
        // supersampled size is ~½ GB the chain would otherwise hold for nothing.
        if !taaStillAccumulation && s.auxA == nil {
            func make32(_ label: String, _ fmt: MTLPixelFormat) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: width, height: height, mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
                let t = device.makeTexture(descriptor: d); t?.label = "Illuminatorama.spec." + label; return t
            }
            guard let cb = make32("colorB", .rgba16Float), let xa = make32("auxA", .rgba32Float), let xb = make32("auxB", .rgba32Float) else { return false }
            s.colorB = cb; s.auxA = xa; s.auxB = xb; s.needsFirstFrame = true
        }
        if s.standIn == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 1, height: 1, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
            s.standIn = device.makeTexture(descriptor: d)
        }
        if s.temporalPipeline == nil {
            let c = engine.pipelineCache
            s.temporalPipeline = c.pipelineState(name: "illumi_spec_temporal", device: device)
            s.atrousPipeline = c.pipelineState(name: "illumi_spec_atrous", device: device)
            s.compositePipeline = c.pipelineState(name: "illumi_spec_composite", device: device)
        }
        guard s.temporalPipeline != nil, s.atrousPipeline != nil, s.compositePipeline != nil else { return false }
        if !s.wasEnabledLastFrame || s.lastStill != taaStillAccumulation || (rtGINeedsFirstFrame && rtGITemporalEnabled) { s.needsFirstFrame = true }
        s.lastStill = taaStillAccumulation
        s.wasEnabledLastFrame = true
        s.writtenThisFrame = true
        return true
    }

    /// Temporal → à-trous → composite. No-op unless the TLAS pass wrote `spec.raw` this frame.
    func encodeRTSpecularDenoise(_ cb: MTLCommandBuffer) {
        let s = rts.spec
        if !s.enabled { s.wasEnabledLastFrame = false }
        guard s.writtenThisFrame, let tp = s.temporalPipeline, let ap = s.atrousPipeline, let cp = s.compositePipeline else { return }
        s.writtenThisFrame = false
        let W = UInt32(width), H = UInt32(height)
        let still = taaStillAccumulation
        var u = SpecDenoiseState.Uniforms(
            width: W, height: H,
            // bit 1: a frozen-camera still averages its frames in the colour resolve, so a temporal mean here would
            // correlate them (the same reasoning as `rtGIStillPerFrame`'s bit 4 for the GI).
            flags: (s.needsFirstFrame ? 1 : 0) | (still ? 2 : 0),
            stepSize: 1,
            sigmaL: s.sigmaL, sigmaZ: s.sigmaZ, sigmaN: s.sigmaN, clampSigma: s.clampSigma,
            mirrorRough: s.mirrorRoughness, fullRough: s.fullRoughness, alphaRough: s.alphaRough,
            maxLevels: UInt32(s.levels),
            // A still averages many independently filtered frames, so a luminance stop that keys on a NOISY
            // neighbour is pure bias (it throws away exactly the bright samples the mean needs — DH-1344's
            // finding for the GI filter): widen its bandwidth floor so only a real edge trips it.
            lumFloor: still ? s.stillLumFloor : 0.02)
        let histC = still ? s.colorA! : (s.toggle ? s.colorA! : s.colorB!)
        let curC = still ? s.colorA! : (s.toggle ? s.colorB! : s.colorA!)
        let histX = still ? s.standIn! : (s.toggle ? s.auxA! : s.auxB!)
        let curX = still ? s.standIn! : (s.toggle ? s.auxB! : s.auxA!)

        if let enc = timedComputeEncoder(cb, "spec.temporal") {
            enc.label = "Illuminatorama.spec.temporal"
            enc.setComputePipelineState(tp)
            enc.setTexture(s.raw, index: 0); enc.setTexture(histC, index: 1); enc.setTexture(histX, index: 2)
            enc.setTexture(gbs.depthTexture, index: 3); enc.setTexture(gbs.gbufferNormalRgh, index: 4)
            enc.setTexture(curC, index: 5); enc.setTexture(curX, index: 6); enc.setTexture(s.varT, index: 7)
            enc.setBytes(&u, length: MemoryLayout<SpecDenoiseState.Uniforms>.stride, index: 0)
            enc.setBuffer(frameUniformBuffer, offset: 0, index: 1)
            dispatch(enc, pipeline: tp, width: width, height: height)
            enc.endEncoding()
        }
        s.toggle.toggle()
        s.needsFirstFrame = false

        // Colour ping-pongs {atrousA, atrousB}; variance {varT → varA → varB → varA …} — disjoint per level.
        var last: MTLTexture = curC
        let levels = max(1, min(5, s.levels))
        for level in 0..<levels {
            u.stepSize = UInt32(1 << level)
            let cIn = level == 0 ? curC : (level % 2 == 1 ? s.atrousA! : s.atrousB!)
            let cOut = level % 2 == 0 ? s.atrousA! : s.atrousB!
            let vIn = level == 0 ? s.varT! : (level % 2 == 1 ? s.varA! : s.varB!)
            let vOut = level % 2 == 0 ? s.varA! : s.varB!
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.label = "Illuminatorama.spec.atrous.\(level)"
            enc.setComputePipelineState(ap)
            enc.setTexture(cIn, index: 0); enc.setTexture(vIn, index: 1)
            enc.setTexture(gbs.depthTexture, index: 2); enc.setTexture(gbs.gbufferNormalRgh, index: 3)
            enc.setTexture(cOut, index: 4); enc.setTexture(vOut, index: 5)
            enc.setBytes(&u, length: MemoryLayout<SpecDenoiseState.Uniforms>.stride, index: 0)
            enc.setBuffer(frameUniformBuffer, offset: 0, index: 1)
            dispatch(enc, pipeline: ap, width: width, height: height)
            enc.endEncoding()
            last = cOut
        }

        if let enc = timedComputeEncoder(cb, "spec.composite") {
            enc.label = "Illuminatorama.spec.composite"
            enc.setComputePipelineState(cp)
            enc.setTexture(last, index: 0); enc.setTexture(pfx.hdrCompositeTexture, index: 1); enc.setTexture(s.fres, index: 2)
            enc.setBytes(&u, length: MemoryLayout<SpecDenoiseState.Uniforms>.stride, index: 0)
            dispatch(enc, pipeline: cp, width: width, height: height)
            enc.endEncoding()
        }
    }
}
