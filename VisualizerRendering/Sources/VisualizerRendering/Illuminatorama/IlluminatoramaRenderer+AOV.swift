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
