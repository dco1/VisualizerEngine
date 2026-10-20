import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - DOFState

/// Depth of field: tile/dilate/prefilter/half/quarter pipelines and targets.
/// Owned by the renderer as `dofs`; every member keeps the name it had on the renderer.
@MainActor
final class DOFState {
    var dofReferenceNeedsReset = true
    var dofReferenceAccumTexture: MTLTexture?

    var dofPipeline: MTLComputePipelineState?
    var dofTilePipeline: MTLComputePipelineState?
    var dofDilatePipeline: MTLComputePipelineState?
    var dofPrefilterPipeline: MTLComputePipelineState?
    /// `dofSharpSubjects`: builds the min-CoC pyramid one level at a time (VZ-0188).
    var dofMinCoCPipeline: MTLComputePipelineState?
    /// `dofFastGather`: the half-resolution background gather.
    var dofHalfPipeline: MTLComputePipelineState?
    /// `dofFastGather`: the max-CoC tile reduction with a threadgroup per tile instead of one
    /// thread walking all 256 texels in series — the same tile map, bit for bit (VZ-0197).
    var dofTileParallelPipeline: MTLComputePipelineState?
    /// `dofQuarterResMinCoC`: the quarter-resolution background gather.
    var dofQuarterPipeline: MTLComputePipelineState?
    /// `lightingDefocusShadingMinCoC`: the least-|CoC|-per-tile reduction the lighting pass gates on,
    /// and its target (r32Float, one texel per 16×16 tile; allocated on first use).
    var dofShadingTilesPipeline: MTLComputePipelineState?
    var lightingRateTileTexture: MTLTexture?
    var dofHalfTexture: MTLTexture?
    /// `dofQuarterResMinCoC`'s target (a quarter of the internal size; allocated on first use).
    var dofQuarterTexture: MTLTexture?
    var dofOutputTexture: MTLTexture?
    var dofTileTexture: MTLTexture?
    var dofTileDilatedTexture: MTLTexture?
    /// Half-resolution, mipmapped, CoC-weighted colour pyramid the gather reads its taps from
    /// (DH-0728/DH-0883). Premultiplied colour with the weight in alpha; the weight is the
    /// texel's own CoC, so in-focus pixels all but vanish from the coarse levels and a sharp
    /// silhouette cannot smear a halo of itself into the bokeh behind it.
    var dofPrefilterTexture: MTLTexture?
    /// `dofSharpSubjects` (VZ-0188): the least |CoC| under each texel of the prefilter pyramid,
    /// same size and levels (r16Float), plus one single-level view per level for its reduction.
    /// The gather asks it whether a tap's prefiltered footprint straddles a sharper layer.
    var dofMinCoCTexture: MTLTexture?
    var dofMinCoCLevels: [MTLTexture] = []
    /// `dofPartialOcclusion`: the NEAR-FIELD coverage pyramid (premultiplied colour + coverage of the
    /// texels in front of the focus plane) and its CoC × coverage, the prefilter's size and mip chain.
    /// Allocated only while the layered composite is on.
    var dofNearTexture: MTLTexture?
    var dofNearCoCTexture: MTLTexture?

    /// True for the current frame when the DOF pass actually ran (so bloom +
    /// tonemap read its output instead of the raw post-FX source).
    var dofApplied: Bool = false
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// Zero the accumulator; takes effect on the next frame's DOF pass.
    public func resetDOFReferenceAccumulation() { dofs.dofReferenceNeedsReset = true }

    // ── Depth-of-field state ─────────────────────────────────────────
    struct DOFParams {
        var invProjection: simd_float4x4
        var focusDist: Float; var cocCoefficient: Float; var maxRadius: Float
        var blades: Float; var bladeRotation: Float; var catsEye: Float
        var width: UInt32; var height: UInt32
        var tileW: UInt32; var tileH: UInt32; var tileSize: UInt32
        /// Mirrors the Metal `DOFParams.prefilterScale`. It used to be declared here as
        /// `_pad0` and never set, which is why the shader's `0 ⇒ point taps` branch was the
        /// only one that ever ran (DH-0883).
        var prefilterScale: Float = 0
        /// Mirrors the Metal `DOFParams.cocFloor` — the diffraction Airy disc's CoC radius in
        /// pixels, applied at every distance including the focus plane (DH-0882).
        var cocFloor: Float = 0
        /// Mirrors the Metal `DOFParams.subjectAware` (1 = `dofSharpSubjects`).
        var subjectAware: Float = 0
        /// Mirrors `DOFParams.fastGather` (1 = `dofFastGather`, with `dofSharpSubjects`).
        var fastGather: Float = 0
        /// Mirrors `DOFParams.halfMinCoC` (`dofHalfResMinCoC`).
        var halfMinCoC: Float = 0
        /// Mirrors `DOFParams.quarterMinCoC` (`dofQuarterResMinCoC`; 0 = no quarter tier).
        var quarterMinCoC: Float = 0
        /// Mirrors `DOFParams.partialOcclusion` (1 = `dofPartialOcclusion`, with `dofSharpSubjects`).
        var partialOcclusion: Float = 0
        /// Mirrors `DOFParams.nearPyramid` (1 = the layered composite's near-field coverage pyramid is
        /// written and read — on whenever `partialOcclusion` is and its textures exist).
        var nearPyramid: Float = 0
    }

    /// Tile size for the max-CoC reduction. Must match nothing in the shader but the
    /// value passed in `DOFParams.tileSize` — the kernel reads it from there.
    static let dofTileSize = 16
}
