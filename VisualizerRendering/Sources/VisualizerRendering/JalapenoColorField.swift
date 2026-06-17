import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

/// Animated GPU metaball colour field for the ChipsAndGuac jalapeños.
///
/// Renders an evaluated metaball potential into a 2D RGBA texture each frame
/// (kernel `jalapenoColorField` in JalapenoColorField.metal). The texture is
/// bound to the jalapeño material's `diffuse.contents` (an `MTLTexture` SceneKit
/// samples directly), so the chilis ripple with a lava-lamp colour shift.
///
/// Per the project's "no faked scrolling textures" rule, the blobs are a real
/// per-pixel metaball field advected over time, not a pre-baked image scrolled
/// across UVs. The Green / Red / Mix toggle is a palette remap of the *same*
/// animated structure.
///
/// Plugs into the shared runtime: takes a `SimEngine`, pulls its pipeline from
/// `SimPipelineCache`, and submits on the engine's command queue. No parallel
/// device stack.
@MainActor
public final class JalapenoColorField {

    public enum Mode: UInt32 { case green = 0, red = 1, mix = 2 }

    private static let log = Logger(subsystem: AppLog.subsystem, category: "JalapenoColorField")

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    public let texture: MTLTexture

    /// Mirrors `JalapenoFieldUniforms` in the metal kernel.
    private struct Uniforms {
        var time: Float
        var speed: Float
        var mode: UInt32
        var blobCount: UInt32
        var texSize: SIMD2<Float>
        var pad: SIMD2<Float>
    }

    public var mode: Mode = .green
    public var speed: Float = 1.0

    private var elapsed: Float = 0
    /// Back-pressure: at most 2 of our colour-field command buffers inflight
    /// so we never block the main actor in `makeCommandBuffer()` if the GPU
    /// falls behind. Drops the occasional field update instead of stalling.
    private let inflight = DispatchSemaphore(value: 2)

    public init?(engine: SimEngine, resolution: Int = 256) {
        self.engine = engine
        guard let pipe = engine.pipeline("jalapenoColorField") else {
            Self.log.error("jalapenoColorField pipeline lookup failed")
            return nil
        }
        self.pipeline = pipe

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: resolution, height: resolution, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        desc.storageMode = .private
        guard let tex = engine.device.makeTexture(descriptor: desc) else {
            Self.log.error("colour-field texture alloc failed")
            return nil
        }
        tex.label = "JalapenoColorField"
        self.texture = tex
    }

    /// Encode one field-update dispatch and commit. Called per visual frame.
    public func tick(dt: Float) {
        elapsed += dt
        // Non-blocking back-pressure acquire — skip this frame's update if
        // the GPU hasn't drained the last two.
        guard inflight.wait(timeout: .now()) == .success else { return }
        guard let cb = engine.commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            inflight.signal()
            return
        }
        cb.label = "JalapenoColorField.tick"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(texture, index: 0)
        var u = Uniforms(
            time: elapsed,
            speed: speed,
            mode: mode.rawValue,
            blobCount: 8,
            texSize: SIMD2(Float(texture.width), Float(texture.height)),
            pad: .zero
        )
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let tg = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(width: texture.width, height: texture.height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.addCompletedHandler { [inflight] _ in inflight.signal() }
        cb.commit()
    }
}
