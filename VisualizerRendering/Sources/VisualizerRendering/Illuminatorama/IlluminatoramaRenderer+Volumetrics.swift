import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - VolumetricState

/// Volumetric shafts, spot-beam scattering and in-view clouds.
/// Owned by the renderer as `vols`; every member keeps the name it had on the renderer.
@MainActor
final class VolumetricState {
    var volRTProgressiveIndex: UInt32 = 0

    var volPipeline: MTLComputePipelineState?
    var volUniformBuffer: MTLBuffer!
    var volFrameSeed: UInt32 = 0

    var volSpotPipeline: MTLComputePipelineState?
    var volSpotUniformBuffer: MTLBuffer!

    var cloudInViewPipeline: MTLComputePipelineState?

    /// Last frame's value, for motion vectors.
    var lastFieldSinkProgress: Float = 0

    var cloudLowResTexture: MTLTexture?
}
