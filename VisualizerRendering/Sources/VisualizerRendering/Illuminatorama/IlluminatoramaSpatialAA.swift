import Foundation
import Metal

/// **Spatial anti-aliasing for a frame with no temporal history** (Daydream, 2026-11-16).
///
/// The live canvas draws one picture — the same while the user drags as when they stop — so it runs
/// with TAA off and no jittered accumulation: any history is something that can ghost, and the
/// ghost on settle is exactly what the canvas gave up. Without a history, and at a working rung
/// below the screen's resolution (0.55×), every edge prints as a staircase. This pass is FXAA 3.11's
/// quality path (`IlluminatoramaSpatialAA.metal`) on the scene-referred HDR at INTERNAL resolution,
/// after the post-resolve overlay and before exposure, DOF, bloom and the tonemap's upscale — the
/// stage where the steps exist. It reads only the current frame, so it cannot smear.
///
/// Off unless the host asks (`IlluminatoramaRenderer.spatialAntiAliasing`); skipped automatically
/// while TAA runs (TAA already anti-aliases, and the export lane's jittered accumulation is a true
/// supersample).
@MainActor
final class IlluminatoramaSpatialAA {
    struct Params { var edgeThreshold: Float; var edgeThresholdMin: Float; var subpix: Float; var pad: Float = 0 }

    static let kernelName = "illumiSpatialAA"
    /// FXAA's "quality" defaults, sub-pixel term at ½: the 0.75 default visibly softens fine texture
    /// (terrazzo, wood grain) on a frame that is already rendered below the screen's resolution.
    static let defaultParams = Params(edgeThreshold: 0.125, edgeThresholdMin: 0.0312, subpix: 0.5)

    var enabled = false
    var params = defaultParams
    /// This frame's anti-aliased frame, or nil when the pass did not run. Reset at the top of `encode`.
    private(set) var appliedOutput: MTLTexture?

    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState?
    private var output: MTLTexture?

    convenience init(engine: SimEngine) {
        self.init(device: engine.device, pipeline: { engine.pipeline($0) })
    }

    init(device: MTLDevice, pipeline: (String) -> MTLComputePipelineState?) {
        self.device = device
        self.pipeline = pipeline(Self.kernelName)
    }

    /// Anti-alias `source` (rgba16Float HDR) into this pass's own target and return it, or nil (nothing
    /// encoded) when off. `appliedOutput` mirrors the return value for the rest of the frame.
    @discardableResult
    func encode(_ cb: MTLCommandBuffer, source: MTLTexture, active: Bool,
                makeEncoder: (String) -> MTLComputeCommandEncoder?) -> MTLTexture? {
        appliedOutput = nil
        guard enabled, active, let pipeline, let out = target(like: source),
              let enc = makeEncoder("spatialAA") else { return nil }
        enc.label = "Illuminatorama.spatialAA"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(source, index: 0)
        enc.setTexture(out, index: 1)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        let tw = pipeline.threadExecutionWidth, th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        enc.dispatchThreads(MTLSize(width: out.width, height: out.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        enc.endEncoding()
        appliedOutput = out
        return out
    }

    private func target(like src: MTLTexture) -> MTLTexture? {
        if let o = output, o.width == src.width, o.height == src.height, o.pixelFormat == src.pixelFormat { return o }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: src.pixelFormat, width: src.width,
                                                         height: src.height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        output = device.makeTexture(descriptor: d)
        output?.label = "Illuminatorama.spatialAA.out"
        return output
    }
}

extension IlluminatoramaRenderer {
    /// **Spatial (FXAA-class) anti-aliasing of the internal-resolution HDR** for hosts that render
    /// without a temporal history. Ignored while TAA is on. Default off — byte-identical frames.
    public var spatialAntiAliasing: Bool {
        get { spatialAAPass.enabled }
        set { spatialAAPass.enabled = newValue }
    }

    /// Encoded right after the post-resolve overlay; `displaySource` then hands its output to the
    /// exposure meter, DOF, the selection outline, bloom and the tonemap.
    func encodeSpatialAA(_ cb: MTLCommandBuffer) {
        spatialAAPass.encode(cb, source: displaySourceBeforeAA, active: !taaEnabled,
                             makeEncoder: { self.timedComputeEncoder(cb, $0) })
    }
}
