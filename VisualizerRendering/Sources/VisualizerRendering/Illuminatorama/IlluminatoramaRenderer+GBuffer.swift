import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - GBufferState

/// G-buffer targets and pipelines, instance buffers, mesh draw groups, scene-stability counters, frame pacing.
/// Owned by the renderer as `gbs`; every member keeps the name it had on the renderer.
@MainActor
final class GBufferState {
    var lastUploadedInstances: [IlluminatoramaRenderer.InstanceRef] = []
    var instanceStableFrames: Int = 0
    /// Frames since any instance's SHAPE (meshKind / modelMatrix / sway fields)
    /// changed — a weaker invariant than full stability that the shadow passes
    /// can gate on, because they never read materials or emission. Always ≥
    /// `instanceStableFrames`.
    var instanceShapeStableFrames: Int = 0

    /// True when any current instance self-oscillates in the vertex shader
    /// (swayMode 2, the top-pivot pendulum, or 3, the wind hinge flap): its silhouette animates with
    /// `time` even though the CPU data is static, so the shadow-skip must not
    /// engage. Modes 0/1 are static functions of per-instance data (a mode-1
    /// lean/jostle change rewrites the instance → resets stability anyway), so
    /// they don't block the skip — sofas with mode-1 cushions still benefit.
    /// Recomputed only when content changes.
    var contentHasSway = false

    /// Tracked by the renderer each frame (set from the host's
    /// `render()` argument). Used as the auto-exposure EMA's `dt`.
    var lastFrameDuration: Double = 1.0 / 60.0

    let gpuMeter = IlluminatoramaGPUMeter()
    // #60 task 4 / #6 — per-compute-pass GPU timer (env-gated). Assigned in init
    // once `device` exists. Off by default; no-ops without VIZ_ILLUMI_PASS_PROFILE.
    var passTimer: IlluminatoramaPassTimer!

    let inFlightSemaphore = DispatchSemaphore(value: IlluminatoramaRenderer.maxFramesInFlight)

    let swapProbePath = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SWAP_PATH"]
    var swapLastTime: CFTimeInterval = 0
    var swapIntervalsMs: [Double] = []

    var gbufferAlbedoMet: MTLTexture!
    var gbufferNormalRgh: MTLTexture!
    var gbufferEmission: MTLTexture!
    /// DH-0140 — the photo-lane sixth G-buffer target (`GBufferOut.material`, RGBA16F). nil until
    /// `extendedGBufferEnabled` first asks for it; dropped by `resize`.
    var gbufferMaterial: MTLTexture?
    var depthTexture: MTLTexture!
    /// Last frame's G-buffer depth, blit-copied from `depthTexture` at the end of
    /// each frame and read by the TAA resolve for disocclusion rejection (so a
    /// fast mover that vacated a pixel onto a similar-luma surface doesn't trail —
    /// the case neighbourhood-clamp alone can't catch). Standalone, not pooled: the
    /// copy is a GPU blit on the frame's own command buffer, and command buffers on
    /// one queue execute in order, so it is ordered against the next frame's G-buffer
    /// regardless of the in-flight pool depth.
    var previousDepthTexture: MTLTexture!
    var hdrTexture: MTLTexture!

    /// Light-layer G-buffer target (R32Uint) — see `Targets.layer`.
    var gbufferLayer: MTLTexture!

    var gbufferPipeline: MTLRenderPipelineState!
    /// Issue #65 — G-buffer pipeline specialized with `kUsePrevVerts = true`. Used
    /// ONLY for the deforming GPU-mesh draws so their `illumi_vs` reads last frame's
    /// per-vertex positions from buffer(5) and writes a real motion vector. Every
    /// other draw keeps `gbufferPipeline` (kUsePrevVerts = false), unchanged.
    var gbufferPipelinePrevVerts: MTLRenderPipelineState!
    /// DH-0140 — the six-target variants (kExtendedGBuffer = true), compiled on first use so the
    /// live app never pays the fragment compile for a lane it may never enter.
    var gbufferPipelineExt: MTLRenderPipelineState?
    var gbufferPipelinePrevVertsExt: MTLRenderPipelineState?
    /// Interior-mapping variants (kInteriorMap = true), built on first use (live / extended lane).
    var gbufferPipelineInterior: MTLRenderPipelineState?
    var gbufferPipelineInteriorExt: MTLRenderPipelineState?

    var depthState: MTLDepthStencilState!
    var lightingPipeline: MTLComputePipelineState!

    var meshGroups: [IlluminatoramaRenderer.MeshDrawGroup] = []

    /// World bounds of each entry in `meshGroups`, index-parallel to it.
    var meshGroupCullBounds: [IlluminatoramaCullBounds] = []

    var cullBoundsKey: IlluminatoramaRenderer.CullBoundsKey?
    /// Bumped whenever `uploadInstances` regroups. NOT bumped on its static-skip frames, whose
    /// grouping is identical — which is what lets the bounds survive them for free.
    var meshGroupsVersion = 0
    /// The exact view-projection the G-buffer pass rasterises with — JITTERED when TAA is. The
    /// cull contract is "the matrix the pass uses", not an unjittered approximation of it.
    var gbufferCullVP = matrix_identity_float4x4

    // ── Perfect analytic superquadric (hero primitive) ─────────────────────────
    // The impostor pipeline (ray-traces the analytic surface in-fragment, writes
    // the G-buffer + analytic depth + motion vectors) and a parallel param buffer.
    // `impostorMeshKinds` are drawn by the impostor pipeline and SKIPPED by the
    // normal raster loop; `rtProxyMeshKinds` are skipped by BOTH raster loops and
    // exist only in the TLAS (RT shadows/GI/reflections). Both sets are populated
    // by `addSuperquadric` (IlluminatoramaSuperquadric.swift) — internal so that
    // extension can mutate them. The instance struct is untouched: raster-exclusion
    // is host-side via these sets, so there is no stride change.
    var superquadricImpostorPipeline: MTLRenderPipelineState?
    /// Ring of `maxFramesInFlight` superquadric-param buffers (see `frameRingIndex`).
    /// Was single-buffered; with `maxFramesInFlight == 2` the previous frame's
    /// impostor draw can still be reading slot N while the CPU writes frame N+1,
    /// so this is now cycled per accepted frame exactly like the instance ping-pong.
    var superquadricParamRing: [MTLBuffer]!

    var superquadricParamCapacity: Int!
    var impostorMeshKinds: Set<IlluminatoramaRenderer.MeshKind> = []

    /// The most recently committed FRAME command buffer (DH-0537 #1). The opt-in / kill-switch
    /// CPU paths that rewrite GPU-read buffers in place (`surfCardBuffer`, `surfCardDirtyBuffer`,
    /// `texelCard`, `cardRect`, `cardRequested`) wait on it first — see
    /// `awaitInFlightFrameBeforeCPUCacheWrite`.
    var lastCommittedFrameCB: MTLCommandBuffer?

    /// **The awaited-frame handshake** — armed by `renderAwaited()`, consumed by
    /// `awaitCommittedFrame()`. See `renderAwaited()` for why a second completion path
    /// exists at all; these two properties are the whole of its state.
    ///
    /// `armAwaitedFrame` is read-and-cleared at the TOP of `render`, so every exit path of
    /// that function — including the empty-scene and dropped-frame early returns — leaves it
    /// disarmed and a later plain `render(blocking:)` cannot inherit it.
    var armAwaitedFrame = false
    /// The signal the next `awaitCommittedFrame()` waits on, or nil when the frame it would
    /// have described never reached the GPU (an early return above). Nil is therefore
    /// "nothing to wait for", not an error.
    var pendingFrameCompletion: FrameCompletionSignal?

    // Phase 2.7 — ping-ponged instance buffers so the G-buffer vertex shader
    // can read both this frame's and last frame's per-instance model matrix
    // at the same `[[instance_id]]`, computing the prev clip-space position
    // for motion-vector reconstruction. `currentInstanceBuffer` /
    // `previousInstanceBuffer` resolve through `useBufferA`.
    //
    // ASSUMPTION (load-bearing): `instances[i]` this frame refers to the
    // same logical instance as `instances[i]` last frame. The vertex shader
    // reads `prevInstances[iid]` to build the previous-frame world position;
    // if the host shuffles the array (or inserts in the middle) between
    // frames, motion vectors will be wrong and TAA will smear wildly.
    //
    // Today this is fine: IlluminatoramaLab pushes a stable [ground, box,
    // sphere] every frame, and any forthcoming scene extractor should
    // similarly produce a stable index per logical SCNNode. The agent's
    // safety-net is `ensureBufferCapacity` setting `taaNeedsFirstFrame =
    // true` on growth — that catches count CHANGES but not REORDERS.
    //
    // When dynamic scene extraction lands, the right fix is a
    // stable per-instance ID (UUID in the host-side struct, hashed into a
    // uint32 in the GPU struct); the renderer can then sort `prevInstances`
    // by ID at the start of each frame so `prevInstances[iid]` lines up
    // with `instances[iid].id`. Don't ship that scene extractor without
    // also wiring this through.
    var instanceBufferA: MTLBuffer!
    var instanceBufferB: MTLBuffer!
    var instanceCapacity: Int!
    var useBufferA: Bool = true

    // One-entry memo of the last resolved lighting variant, so the steady-state
    // path (flags unchanged frame-to-frame, the overwhelmingly common case) is a
    // single UInt8 compare with zero allocation — no MTLFunctionConstantValues or
    // key string built. 0xFF is an impossible 6-bit value, so the first call misses.
    var lastLightingFlags: UInt8 = 0xFF
    var lastLightingPipeline: MTLComputePipelineState?

    /// One-entry memo of `lightingShadingRatePipelines` (flags as `currentLightingPipeline` packs them).
    var lastRatePipelines: (flags: UInt8, pair: (coarse: MTLComputePipelineState, skip: MTLComputePipelineState)?) = (0xFF, nil)

    /// `MeshKind` → its contiguous `[start, end)` slot range in the grouped
    /// instance buffer. Built alongside `meshGroups` in `uploadInstances` (and
    /// preserved across static-skip frames, whose grouping is identical) so the
    /// per-frame `onEncodeGPUInstances` lookup is a hash probe, not a linear
    /// scan of `meshGroups`.
    var meshGroupRange: [IlluminatoramaRenderer.MeshKind: Range<Int>] = [:]
}
