import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - GlassState

/// Glass (forward RT / fallback) and caustics.
/// Owned by the renderer as `gls`; every member keeps the name it had on the renderer.
@MainActor
final class GlassState {
    var lastGlassFlat: [(kind: IlluminatoramaRenderer.MeshKind, insts: [IlluminatoramaGlassInstance])] = []
    var glassStableFrames: Int = 0

    var glassRTPipeline: MTLRenderPipelineState?  // illumi_glass_rt_fs (traces TLAS)
    var glassFallbackPipeline: MTLRenderPipelineState?  // illumi_glass_fallback_fs (Fresnel+sky)
    var glassDepthState: MTLDepthStencilState?
    /// `rtGlassDefocusTraceFactor` targets: the reduced-resolution trace (rgba16Float, coverage
    /// in alpha) and a same-size depth target cleared to far — the RT glass pipeline carries a
    /// depth attachment, and a cleared one passes every fragment (no occlusion test at low res).
    /// Allocated on first use; nil for every scene that never opts in.
    var glassDefocusTexture: MTLTexture?
    var glassDefocusDepth: MTLTexture?

    /// Per-glass-TLAS-instance material (RTGlassData[], grouped == TLAS glass
    /// order), rebuilt with the TLAS. Indexed by `instance_id - glassInstanceBase`.
    var rtGlassDataBuffer: MTLBuffer?
    /// First glass instance_id in the TLAS (= opaque count + curve count). The RT
    /// glass shader uses this to tell a glass hit from an opaque one.
    var rtGlassInstanceBase: Int = 0
    var rtGlassInstanceCount: Int = 0

    var causticDecayPipeline: MTLComputePipelineState?
    var causticPhotonsPipeline: MTLComputePipelineState?
    var causticPrimaryPipeline: MTLComputePipelineState?
    /// `atomic_float` RGB atlas (3 floats/texel), sized to the surface-cache atlas.
    var causticAtlasBuffer: MTLBuffer?
    /// Per-glass-instance emitter discs (float4 centre+radius), rebuilt per frame.
    var causticDiscBuffer: MTLBuffer?
    var causticAtlasTexels: Int = 0

    var causticFrameSeed: UInt32 = 0

    /// Pre-glass copy of `hdrCompositeTexture`, allocated lazily only when cheap
    /// screen-space glass (mode 2) is in use. The glass pass samples this for the
    /// scene behind the pane (you can't sample a render target you're also writing).
    var glassBackdropTexture: MTLTexture?

    /// Concatenated per-corner SHADING normals of the meshes glass instances use (3 × `float4`
    /// per triangle). A glass instance's slice starts at `RTInstanceData.emissionPad.w − 1`
    /// (0 = none → the bounce loop keeps the face normal). Glass-only on purpose: opaque hits
    /// shade with the face normal and would triple this buffer for nothing.
    var rtGlassCornerNormalBuffer: MTLBuffer?

    /// Per-frame memo for `flattenGlass()` — nil = not yet flattened this frame.
    var glassFlatMemo: [(kind: IlluminatoramaRenderer.MeshKind, insts: [IlluminatoramaGlassInstance])]?
}
