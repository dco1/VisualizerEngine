import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - SelectionState

/// Selection outline pass: mask, dilation, composite.
/// Owned by the renderer as `outl`; every member keeps the name it had on the renderer.
@MainActor
final class SelectionState {
    // ── Highlight outline pipeline state ─────────────────────────────
    var selectionBoxMaskPipeline: MTLRenderPipelineState?  // mask boxes, filtered by wantMode
    var selectionDilateHPipeline: MTLComputePipelineState?  // horizontal max-filter
    var selectionDilateVPipeline: MTLComputePipelineState?  // vertical max-filter
    var selectionCompositePipeline: MTLComputePipelineState?  // ring → HDR additive composite
    var selectionMaskTexture: MTLTexture?  // 0/1 per pixel (mask boxes rendered here)
    var selectionDilateHTexture: MTLTexture?  // horizontal-dilated intermediate
    var selectionDilatedTexture: MTLTexture?  // fully dilated result
    var selectionMaskInstanceBuffer: MTLBuffer?  // GPU copy of `highlightMaskInstances`
    var selectionDepthState: MTLDepthStencilState?
}
