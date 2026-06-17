import Foundation
import Metal
import OSLog
import QuartzCore
import VisualizerCore

/// Generates an animated kaleidoscope equirect HDR texture on the GPU.
///
/// Drop-in source for an Illuminatorama `equirectSky`: the pattern both fills
/// the background (sky pixels) and drives IBL (irradiance + prefiltered specular
/// cubes baked from this texture). Used by Egg Roller Ultra's "Kaleidoscope
/// Background" mode, where the warehouse shell is removed and the eggs + rails
/// float in a turning kaleidoscope.
///
/// One compute dispatch per `render(time:)` writes the whole equirect — no CPU
/// readback, no per-frame allocation. The output is `.rgba16Float` so petals can
/// exceed 1.0 and feed HDR bloom.
@MainActor
public final class KaleidoscopeSky {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "kaleidoscopeSky")

    public private(set) var outputTexture: MTLTexture

    /// Mirrored wedge count. 6–12 reads as a classic kaleidoscope.
    public var segments: Float = 8
    /// HDR gain on the output colour. >1 makes the brightest petals bloom and
    /// pushes more energy into the baked IBL.
    public var gain: Float = 1.6

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState?
    private let uniformsBuffer: MTLBuffer

    private struct Uniforms {
        var time: Float
        var segments: Float
        var gain: Float
        var _pad: Float = 0
    }

    public init(engine: SimEngine = .shared,
                resolution: SIMD2<Int> = SIMD2<Int>(1024, 512)) {
        precondition(resolution.x > 0 && resolution.y > 0, "Bad kaleidoscope resolution")
        self.device = engine.device
        self.queue = engine.commandQueue
        self.outputTexture = Self.makeTexture(device: engine.device,
                                              width: resolution.x,
                                              height: resolution.y,
                                              label: "KaleidoscopeSky.equirect")
        self.pipeline = engine.pipelineCache.pipelineState(name: "kaleidoscope_sky",
                                                           device: engine.device)
        guard let buf = device.makeBuffer(length: MemoryLayout<Uniforms>.stride,
                                          options: .storageModeShared) else {
            preconditionFailure("KaleidoscopeSky: uniforms buffer alloc failed")
        }
        buf.label = "KaleidoscopeSky.uniforms"
        self.uniformsBuffer = buf
        if pipeline == nil {
            Self.log.error("kaleidoscope_sky pipeline missing — check Shaders/KaleidoscopeSky.metal compiles")
        }
    }

    /// Dispatch one frame of the kaleidoscope into `outputTexture`.
    public func render(time: Float) {
        guard let pipeline else { return }
        var u = Uniforms(time: time, segments: segments, gain: gain)
        uniformsBuffer.contents().copyMemory(from: &u, byteCount: MemoryLayout<Uniforms>.stride)

        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        cb.label = "KaleidoscopeSky.render"
        enc.label = "KaleidoscopeSky.dispatch"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(outputTexture, index: 0)
        enc.setBuffer(uniformsBuffer, offset: 0, index: 0)

        let w = pipeline.threadExecutionWidth
        let hgt = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let tg = MTLSize(width: w, height: hgt, depth: 1)
        let grid = MTLSize(width: (outputTexture.width  + w   - 1) / w,
                           height: (outputTexture.height + hgt - 1) / hgt,
                           depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
    }

    private static func makeTexture(device: MTLDevice,
                                    width: Int, height: Int,
                                    label: String) -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.pixelFormat = .rgba16Float
        d.width = width
        d.height = height
        d.usage = [.shaderWrite, .shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            preconditionFailure("KaleidoscopeSky: makeTexture failed")
        }
        t.label = label
        return t
    }
}
