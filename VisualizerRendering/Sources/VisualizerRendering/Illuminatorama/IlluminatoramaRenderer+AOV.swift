import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - AOVState

/// Per-term AOV capture.
/// Owned by the renderer as `aovs`; every member keeps the name it had on the renderer.
@MainActor
final class AOVState {
    var aovTexture: MTLTexture?
    var aovResetPending = true
    var aovActiveThisFrame = false
    var aovFrameBlend: Float = 1
    var rtTLASAOVPipeline: MTLComputePipelineState?
    var rtTLASAOVCurvePipeline: MTLComputePipelineState?
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// Restart the AOV running mean on the next captured frame.
    public func resetAOVAccumulation() { aovs.aovResetPending = true }

    /// Mirror of `AOVUniforms` in IlluminatoramaAOV.h (16 B).
    struct AOVUniforms { var blend: Float; var handoff: UInt32; var pad0: UInt32 = 0; var pad1: UInt32 = 0 }

    /// Decide this frame's capture BEFORE the lighting pass: allocate the means at the internal size,
    /// restart them when asked or when the GI mean restarts, and fix the frame's 1/N.
    func prepareAOVFrame() {
        guard aovCaptureEnabled else { aovs.aovActiveThisFrame = false; return }
        if let t = aovs.aovTexture, t.width == width, t.height == height {
        } else {
            let d = MTLTextureDescriptor()
            d.textureType = .type2DArray
            d.pixelFormat = .rgba32Float
            d.width = width; d.height = height
            d.arrayLength = IlluminatoramaAOV.sliceCount
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            aovs.aovTexture = device.makeTexture(descriptor: d)
            aovs.aovTexture?.label = "Illuminatorama.aov"
            aovs.aovResetPending = true
        }
        guard aovs.aovTexture != nil else { aovs.aovActiveThisFrame = false; return }
        // The GI mean's restart is honoured only when that flag will be consumed this frame (with the
        // GI accumulator off it stays raised forever, which would restart this mean every frame).
        let giRestarts = rtGINeedsFirstFrame && rtGITemporalEnabled && rtEnabled && rtSupported
        if aovs.aovResetPending || giRestarts { aovAccumulatedFrames = 0; aovs.aovResetPending = false }
        aovAccumulatedFrames += 1
        aovs.aovFrameBlend = 1 / Float(aovAccumulatedFrames)
        aovs.aovActiveThisFrame = true
    }

    /// The `kRTAOVEnabled` variant of the TLAS lighting kernel (with / without curves), compiled
    /// BLOCKING on first use — a capture lane, never a vsync one.
    func rtTLASAOVVariant(curves: Bool) -> MTLComputePipelineState? {
        if curves, let p = aovs.rtTLASAOVCurvePipeline { return p }
        if !curves, let p = aovs.rtTLASAOVPipeline { return p }
        let cv = MTLFunctionConstantValues()
        var on = true
        cv.setConstantValue(&on, type: .bool, index: Self.rtAOVFunctionConstantIndex)
        if curves { cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex) }
        let p = engine.pipelineCache.pipelineState(name: "illumi_rt_lighting_tlas", device: device,
                                                   constants: cv, variantKey: curves ? "aov-curves" : "aov")
        if curves { aovs.rtTLASAOVCurvePipeline = p } else { aovs.rtTLASAOVPipeline = p }
        return p
    }

    /// One slice of the captured means, read back as rgba32Float rows (`width × height × 4`).
    /// Waits for the GPU — diagnostics only.
    public func readAOV(_ aov: IlluminatoramaAOV) -> (width: Int, height: Int, rgba: [Float])? {
        guard let slice = aov.slice, let tex = aovs.aovTexture else { return nil }
        let w = tex.width, h = tex.height
        let rowBytes = w * MemoryLayout<Float>.stride * 4
        guard let buf = device.makeBuffer(length: rowBytes * h, options: .storageModeShared),
              let cb = commandQueue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: tex, sourceSlice: slice, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: buf, destinationOffset: 0, destinationBytesPerRow: rowBytes, destinationBytesPerImage: rowBytes * h)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: diagnostics readback, never the per-frame lane
        let n = w * h * 4
        let out = [Float](unsafeUninitializedCapacity: n) { dst, count in
            memcpy(dst.baseAddress!, buf.contents(), n * MemoryLayout<Float>.stride)
            count = n
        }
        return (w, h, out)
    }
}
