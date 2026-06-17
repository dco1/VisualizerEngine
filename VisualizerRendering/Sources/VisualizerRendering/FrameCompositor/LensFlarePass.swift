import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── LENS FLARE PASS ─────────────────────────────────────────────────────────
//
// Screen-space additive lens flare. Reads `BurstLightField`, projects each
// burst's world position to NDC via the view-projection matrix, paints
// anamorphic streak + ghost dots into the compositor's HDR colour
// attachment. Runs BEFORE bloom + tonemap so the flare is naturally
// smeared by the bloom chain.
//
// REUSABLE for any scene that wants a cinematic "camera responds to
// bright sources" look. Pair with a per-burst registration into
// BurstLightField (already shared infrastructure).

@MainActor
public final class LensFlarePass {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "LensFlarePass")

    public let device: MTLDevice
    public let engine: SimEngine
    public let burstField: BurstLightField

    private let pipeline: MTLComputePipelineState
    private let uniformBuffer: MTLBuffer

    public var globalGain: Float = 0.6
    public var anamorphicStretch: Float = 3.5
    public var ghostIntensity: Float = 0.45

    public init?(engine: SimEngine, burstField: BurstLightField) {
        guard let pipeline = engine.pipeline("lensFlarePass") else {
            Self.log.error("lensFlarePass pipeline lookup failed")
            return nil
        }
        let device = engine.device
        guard let uBuf = device.makeBuffer(
            length: MemoryLayout<LensFlareUniforms>.stride,
            options: .storageModeShared
        ) else {
            Self.log.error("LensFlarePass uniform buffer alloc failed")
            return nil
        }
        uBuf.label = "LensFlarePass.uniforms"

        self.device = device
        self.engine = engine
        self.burstField = burstField
        self.pipeline = pipeline
        self.uniformBuffer = uBuf
    }

    public func encode(
        into cb: MTLCommandBuffer,
        target: FrameCompositor,
        viewProjection: float4x4
    ) {
        let count = UInt32(burstField.activeCount)
        guard count > 0 else { return }
        let countAsFloat = Float(bitPattern: count)

        // Pack rows of view-projection for the kernel's dot products.
        let row0 = SIMD4<Float>(viewProjection.columns.0.x, viewProjection.columns.1.x,
                                viewProjection.columns.2.x, viewProjection.columns.3.x)
        let row1 = SIMD4<Float>(viewProjection.columns.0.y, viewProjection.columns.1.y,
                                viewProjection.columns.2.y, viewProjection.columns.3.y)
        let row2 = SIMD4<Float>(viewProjection.columns.0.z, viewProjection.columns.1.z,
                                viewProjection.columns.2.z, viewProjection.columns.3.z)
        let row3 = SIMD4<Float>(viewProjection.columns.0.w, viewProjection.columns.1.w,
                                viewProjection.columns.2.w, viewProjection.columns.3.w)

        var u = LensFlareUniforms(
            viewProjRow0: row0,
            viewProjRow1: row1,
            viewProjRow2: row2,
            viewProjRow3: row3,
            countAndGain: SIMD4(countAsFloat, globalGain, anamorphicStretch, ghostIntensity)
        )
        uniformBuffer.contents().copyMemory(
            from: &u, byteCount: MemoryLayout<LensFlareUniforms>.stride
        )

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "LensFlarePass"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(target.hdrColorTexture, index: 0)
        enc.setBuffer(burstField.buffer.buffer, offset: 0, index: 0)
        enc.setBuffer(uniformBuffer,            offset: 0, index: 1)

        let w = target.hdrColorTexture.width
        let h = target.hdrColorTexture.height
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let grid = MTLSize(width: w, height: h, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

struct LensFlareUniforms {
    var viewProjRow0: SIMD4<Float>
    var viewProjRow1: SIMD4<Float>
    var viewProjRow2: SIMD4<Float>
    var viewProjRow3: SIMD4<Float>
    var countAndGain: SIMD4<Float>
}
