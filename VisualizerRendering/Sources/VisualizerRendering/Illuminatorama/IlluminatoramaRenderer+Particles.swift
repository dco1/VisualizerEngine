import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - ParticleState

/// Particles, streaks, and GPU mesh repack / normal-synthesis pipelines.
/// Owned by the renderer as `fxs`; every member keeps the name it had on the renderer.
@MainActor
final class ParticleState {
    // Phase 4.11 — particle pipelines. The compute step integrates
    // positions/velocities/life; the render pipeline draws survivors as
    // additive HDR point sprites into the composite.
    var particleStepPipeline: MTLComputePipelineState!
    var particleDrawPipeline: MTLRenderPipelineState!
    /// Phase 4.23 — host-buffer point-sprite renderer. Stride-aware so
    /// FireworksUltra's `SIMD4<Float>` star buffers + `packed_float3`
    /// burst buffers both bind without a copy.
    var extParticleDrawPipeline: MTLRenderPipelineState!
    // Phase 4.13a — repack kernel for the DynamicMesh bridge.
    var repackPosNormPipeline: MTLComputePipelineState!

    // Phase 4.21 — one-shot GPU mesh normal/tangent synthesis (replaces the
    // CPU passes that ran on the main thread inside IlluminatoramaMesh.from).
    var meshBuildAdjacencyPipeline: MTLComputePipelineState!
    var meshSynthPipeline: MTLComputePipelineState!

    /// Depth-stencil that READS the G-buffer depth without writing.
    /// Opaque geometry occludes particles behind it; particles don't
    /// occlude each other (additive HDR needs every drawn fragment to
    /// add to the composite).
    var particleDepthState: MTLDepthStencilState!

    /// Active particle emitters. Hosts register via `addEmitter` and the
    /// renderer ticks + draws each enabled one per frame.
    var particleEmitters: [IlluminatoramaParticleEmitter] = []
    /// Particle fields published through `SimEngine.particleFields` for the
    /// active scene, refreshed each frame by the extractor via
    /// `setParticleFields`. Each is drawn as additive HDR point sprites into
    /// the same composite the lighting pass writes; bloom + tonemap process
    /// them naturally. (Replaces the Phase 4.23 SCNGeometry-associated-object
    /// bridge with the shared SimEngine registry — see `ParticleFieldSource`.)
    var particleFields: [ParticleFieldSource] = []
    /// Velocity-aligned streak sources refreshed each frame by the host.
    /// Rendered after point-sprite fields in `encodePostResolveFX` so both
    /// participate in the same HDR bloom + tonemap pass.
    var particleStreaks: [IlluminatoramaRenderer.StreakSource] = []
    /// Pipeline for velocity-aligned billboard streak quads (`particleStreakVS/FS`).
    /// `nil` when the shaders are absent from the library — streak draws are
    /// silently skipped in that case (logged once at init).
    var streakPipeline: MTLRenderPipelineState?

    /// Wall-clock of the previous `render()` call, used to derive a real
    /// `dt` for particle integration even when the host's frame rate
    /// varies.
    var lastParticleTickTime: CFTimeInterval = CACurrentMediaTime()

    var gpuRepackTasks: [IlluminatoramaRenderer.GPURepackTask] = []
}
