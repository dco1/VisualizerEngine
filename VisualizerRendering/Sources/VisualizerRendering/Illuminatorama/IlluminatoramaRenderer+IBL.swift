import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - IBLState

/// Image-based lighting: irradiance/prefilter cubes, DFG + LTC tables, sky handoff textures.
/// Owned by the renderer as `ibls`; every member keeps the name it had on the renderer.
@MainActor
final class IBLState {
    /// **DH-0896 — the deferred composite's specular-IBL share, handed to the RT reflection
    /// pass.** `outHDR` already carries the sky cube reflected along R; a reflection ray that
    /// HITS the scene must replace that sky, not be added on top of it (which is what made a
    /// wall mirror read as a sunset with the room ghosted faintly through it). Allocated
    /// lazily at the HDR size the first frame RT reflections run, dropped on resize.
    var specIBLTexture: MTLTexture?

    /// Whether THIS frame's lighting pass wrote `specIBLTexture` — the RT kernel's read is
    /// gated on it, so a frame that skipped the write can never subtract a stale sky.
    var specIBLWrittenThisFrame = false

    /// The deferred composite's diffuse SKY share, handed to the traced GI pass (see
    /// `rtGIReplacesDiffuseSky`). Lazily allocated at the HDR size, dropped on resize.
    var diffSkyTexture: MTLTexture?

    /// Whether THIS frame's lighting pass wrote `diffSkyTexture` — the traced pass only
    /// subtracts a share that was written this frame.
    var diffSkyWrittenThisFrame = false

    // Phase 3 — sky-probe IBL. The irradiance + prefiltered cubes are baked
    // from `equirectSky` each frame (or whenever the sky texture changes).
    var irradianceCube: MTLTexture!
    var prefilteredCube: MTLTexture!
    var prefilteredMipViews: [MTLTexture]!
    /// 1×1 RGBA16F texture used in place of `equirectSky` when the host
    /// hasn't supplied one. Means the bake kernels and lighting pass can
    /// always bind a real texture and never branch on nil.
    var dummySkyTexture: MTLTexture!
    // Phase 3.2 — split-sum DFG LUT. RG16F keyed on (NdotV, roughness),
    // baked once on init via the `illumi_dfg_bake` kernel. View / material /
    // scene-independent so it never rebakes. Texture is bound at slot 10 of
    // the lighting kernel; the `dfgLUTEnabled` knob above gates sampling so
    // we can A/B against the Lagarde fallback without recompiling.
    var dfgLUT: MTLTexture!

    // #60 task 5 increment 2 — LTC area-light specular LUTs (Minv + magnitude),
    // baked once on init. `ltcValidated` is true only when the bake matched
    // brute-force ground truth; the lighting kernel falls back to MRP otherwise.
    // Bound at slots 16/17 of the lighting kernel.
    var ltcMatTexture: MTLTexture!
    var ltcMagTexture: MTLTexture!
    var ltcValidated: Bool!

    /// Wall-clock time (in renderer `time` seconds) of the last successful
    /// IBL bake — used for the `iblRebakeInterval` fallback.
    var iblLastBakeTime: Float = -.infinity

    var irradianceBakePipeline: MTLComputePipelineState!
    var prefilterBakePipeline: MTLComputePipelineState!
    var dfgBakePipeline: MTLComputePipelineState!
}
