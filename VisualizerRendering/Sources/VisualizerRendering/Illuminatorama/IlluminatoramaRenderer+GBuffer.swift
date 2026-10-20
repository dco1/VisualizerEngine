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

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// The shipped default, named once so `init` (which cannot read the
    /// memberwise default before `self` is formed) and the property declaration
    /// can't drift apart.
    public nonisolated static let defaultInternalRenderScale: Float = 1.5

    /// Floor for `internalRenderScale`. **0.5** — a half-linear (quarter-pixel)
    /// internal frame is the lowest tier that still resolves architectural
    /// detail through the tonemap's 4-tap bilinear magnification, and it is the
    /// point where the derived chains bottom out usefully: SSAO/bloom run at
    /// internal/2 and halation at internal/2, so 0.5 of a 1280×800 canvas is
    /// still a 320×200 halation buffer. Below this the derived chains degenerate
    /// toward single-texel buffers, TAA/SSR reprojection has too few samples to
    /// converge, and the upscale is mush — a worse trade than simply drawing
    /// less. Nothing in the pipeline *crashes* below 0.5 (every derived size is
    /// `max(1, …)`-guarded and `internalDims` floors at 2×2), so this is a
    /// quality floor, not a safety one.
    public nonisolated static let minInternalRenderScale: Float = 0.5

    /// Above ~4× the internal targets alone run to gigabytes and every pass is
    /// bandwidth-starved; the ceiling exists so a mis-typed slider can't try to
    /// allocate a 16× frame and fail every texture.
    public nonisolated static let maxInternalRenderScale: Float = 4.0

    // ── Canvas budget ────────────────────────────────────────────────

    /// A hard ceiling on how big a canvas this process is allowed to render,
    /// honoured **at construction** rather than applied afterwards.
    ///
    /// Scenes ship a *fixed* canvas chosen on the machine they were authored on
    /// (`renderWidth`/`renderHeight`, often 2880×1620) and some pin their own
    /// SSAA on top. A host running on a much smaller GPU — the Apple TV app is
    /// the motivating case — can shrink a scene after `make()` returns via
    /// `IlluminatoramaCanvasScalable.setCanvasScale`, but by then the targets
    /// for the FULL canvas have already been allocated: at the documented ~194
    /// bytes per internal pixel, a 2880×1620 canvas at SSAA 1.3 is ~1.5 GB
    /// asked for in one breath. A host with a per-process memory limit is dead
    /// before it can trim anything.
    ///
    /// So the ceiling lives here, where every path that sizes a target already
    /// funnels through: `init`, `resize`, and the `internalRenderScale`
    /// observer. Aspect ratio is always preserved — a clamped canvas is the
    /// same framing at lower resolution, never a crop.
    ///
    /// **Default is unlimited, so this changes nothing unless a host opts in.**
    /// Set it once at launch, before the first renderer is constructed; setting
    /// it later only affects renderers built or resized after the fact.
    public struct CanvasBudget: Sendable, Equatable {
        /// Widest output canvas allowed, in pixels. `nil` = unlimited.
        public var maxOutputWidth: Int?
        /// Highest `internalRenderScale` allowed. `nil` = the usual 4.0 ceiling.
        public var maxInternalRenderScale: Float?

        public init(maxOutputWidth: Int? = nil, maxInternalRenderScale: Float? = nil) {
            self.maxOutputWidth = maxOutputWidth
            self.maxInternalRenderScale = maxInternalRenderScale
        }

        /// The unconstrained default — what every host gets unless it opts in.
        public static let unlimited = CanvasBudget()
    }

    /// Apply `canvasBudget` to one (output size, SSAA) pair.
    ///
    /// Returns the sizes actually allowed. Width is clamped first and the height
    /// follows from the ORIGINAL aspect (not from the clamped width's rounding),
    /// so repeated clamping is stable and framing is preserved exactly.
    static func budgeted(outputW: Int, outputH: Int,
                         scale: Float) -> (w: Int, h: Int, scale: Float) {
        var w = max(1, outputW)
        var h = max(1, outputH)
        let budget = canvasBudget
        if let maxW = budget.maxOutputWidth, w > maxW, maxW > 0 {
            let aspect = Double(w) / Double(max(1, h))
            w = max(2, (maxW / 2) * 2)
            h = max(2, (Int((Double(w) / aspect).rounded()) / 2) * 2)
        }
        var s = scale
        if let maxS = budget.maxInternalRenderScale { s = min(s, maxS) }
        return (w, h, s)
    }

    // ── Frame throttle / pipelining (command-queue saturation guard) ────────────
    //
    // The overlay ticks `render()` at 60 Hz and commits a command buffer every
    // tick. A heavy scene whose GPU frame costs more than the tick interval
    // would otherwise back the queue up until `makeCommandBuffer()` blocks the
    // main thread on Metal's slot semaphore — multi-second freezes, and (the
    // symptom that masqueraded as a white frame) the presented buffer never
    // settling on a completed frame for SceneKit to sample. Cap frames in
    // flight and DROP a tick when the GPU hasn't caught up, rather than block.
    //
    // `maxFramesInFlight = 2` (was 1) is the "live-frame gap" fix. With ONE
    // frame in flight, the next frame can't begin encoding until the previous
    // one fully completes on the GPU; the 60 Hz timer then only starts it on the
    // following tick edge, so a >16.6 ms GPU frame leaves the GPU IDLE from its
    // completion until the next tick fires (e.g. a 24 ms frame finishes at 24 ms
    // but the next render() doesn't start until t=33 ms → ~9 ms wasted → the
    // new-frame rate quantises to 60/ceil(gpuMs/16.6) ≈ 30 fps even though the
    // GPU only needs ~24 ms). DOUBLE-buffering lets tick N+1 encode + queue the
    // next frame while frame N is still on the GPU, so the GPU runs back-to-back
    // and the new-frame rate rises to the GPU-bound ceiling (~1000/gpuMs). This
    // is the standard CPU/GPU pipelining every realtime renderer uses; it is NOT
    // the same as removing the throttle (we still DROP when 2 are already in
    // flight, so the queue stays bounded — no unbounded-growth freeze).
    //
    // Pairs with the `ldrPoolCount = 4` present pool: presented + previously-
    // presented (both possibly still read by SceneKit / the cover blit) + the
    // TWO buffers that may be in flight at once. `acquireWriteTarget()` excludes
    // all four so a frame never writes a buffer another consumer is reading.
    nonisolated static let maxFramesInFlight = 2

    // ── Per-frame buffer-race probe (env-gated diagnostic) ──────────────────────
    // The blocky-lighting / jitter race (see docs/known-issues/illuminatorama-
    // inflight-buffer-race.md) is invisible to the headless harness because the
    // snapshot reads a SETTLED frame — but the race CONDITION still occurs in the
    // headless tick loop. This probe counts it quantitatively, so the bug is now
    // headless-detectable. `VIZ_ILLUMI_RACE_PROBE=1` turns it on; it writes a
    // running `raceEvents=N frames=M` to `VIZ_ILLUMI_RACE_PATH`. With the ring fix
    // the count is 0. `VIZ_ILLUMI_RACE_PIN0=1` pins the ring to slot 0 (reproduces
    // the pre-fix single-buffer behaviour) so the count goes positive — the
    // negative control that proves both the bug and the fix without a live window.
    nonisolated static let raceProbeEnabled = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RACE_PROBE"] == "1"

    nonisolated static let raceProbePin0    = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RACE_PIN0"] == "1"

    // ── Swap-count jitter probe (env-gated; issue #65 "swap-count jitter probe") ──
    // Measures REAL frame-delivery jitter, not the timer cadence the in-tick
    // VIZ_EGG_TRACE measures. Every time `promoteCompletedBuffer` actually swaps
    // `outputTexture` a new fully-rendered frame has been delivered; we timestamp
    // those swaps and histogram the inter-swap intervals into p50/p95/p99. A tight
    // p99-near-p50 = smooth delivery; a long tail = the jitter. `VIZ_ILLUMI_SWAP_PROBE=1`
    // turns it on, writing `swaps=… fps=… p50=… p95=… p99=… max=… (ms)` to
    // `VIZ_ILLUMI_SWAP_PATH`. Touched only on the main actor (in promoteCompletedBuffer).
    nonisolated static let swapProbeEnabled = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SWAP_PROBE"] == "1"

    // Issue #65 — velocity / motion-vector debug overlay. When set, the tonemap is
    // fed a NEGATIVE motionBlurStrength sentinel and renders the colourised velocity
    // buffer instead of the scene (the "per-object velocity overlay" validator).
    nonisolated static let mvDebugOverlay = ProcessInfo.processInfo.environment["VIZ_ILLUMI_MV_DEBUG"] == "1"

    /// Phase 4.12 — GPU-instanced draw recipe built once per frame from
    /// the host's `instances` array. Each entry is one `instanceCount`-
    /// wide draw call: `start` is the offset (in `IlluminatoramaInstance`
    /// units) into both the current and previous instance buffers, and
    /// `count` is the number of instances of `kind` to render in that
    /// single draw call. Built in `uploadInstances` after grouping by
    /// mesh kind; consumed by every pass that iterates the scene's
    /// per-instance data (G-buffer, cascaded shadows, spot shadows).
    struct MeshDrawGroup {
        let kind: MeshKind
        let start: Int
        let count: Int
    }

    /// What `meshGroupCullBounds` was built from. Sway bounds are angle-independent by
    /// construction (the sphere about the pivot), so TIME is deliberately not in here — only a
    /// regroup, a wind-strength change, or a change in what the GPU is allowed to overwrite.
    struct CullBoundsKey: Equatable {
        var groupsVersion: Int
        var wind: Float
        var hostWritesInstances: Bool
        var repackCount: Int
    }

    public func setMesh(_ kind: MeshKind, _ mesh: IlluminatoramaMesh) {
        meshes[kind] = mesh
    }

    /// Dispatch a 1-D compute grid covering `count` threads, threadgroup sized
    /// to the pipeline's execution width.
    func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            count: Int) {
        let tgWidth = max(1, min(pipeline.maxTotalThreadsPerThreadgroup,
                                 pipeline.threadExecutionWidth * 8))
        let groups = (count + tgWidth - 1) / tgWidth
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    }

    // ── Phase 4.8 — GPU-direct mesh registration ───────────────────────
    //
    // Compute-fed callers (`DynamicMesh`, `BarkRenderer`, MLS-MPM solvers)
    // hand the renderer a ready-made `IlluminatoramaMesh` whose vertex /
    // index buffers they own. The renderer stores it in `meshes` under a
    // freshly-generated `MeshKind.custom(UUID)` so each registration is
    // unique; the returned `IlluminatoramaMeshHandle` carries the kind
    // back to the host, who attaches it to the relevant `SCNGeometry` via
    // `SCNGeometry.illuminatoramaMeshHandle`. The scene extractor checks
    // the associated object before falling through to the CPU-readback
    // path, so a single attach is all the wiring needed per geometry.

    @discardableResult
    /// Object-space AABB of a registered mesh, or `nil` if it cannot be read.
    /// Diagnostic path — see `IlluminatoramaMesh.localBounds()`.
    public func meshLocalBounds(_ kind: MeshKind) -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        meshes[kind]?.localBounds()
    }

    public func registerMesh(_ mesh: IlluminatoramaMesh) -> IlluminatoramaMeshHandle {
        let kind = MeshKind.custom("gpuMesh#\(UUID().uuidString)")
        meshes[kind] = mesh
        return IlluminatoramaMeshHandle(kind: kind, mesh: mesh, renderer: self)
    }

    /// The registered mesh behind a `MeshKind`, or nil if nothing is registered under it.
    ///
    /// Read-only lookup into the mesh table, for a host that needs the geometry it is DRAWING
    /// rather than a copy of what it once submitted — exact cursor picking (ray vs the real
    /// triangles, via `IlluminatoramaMesh.objectTriangleSoup()`), a measurement probe, a
    /// geometry audit. The alternative is for every host to shadow the table with its own
    /// `[MeshKind: IlluminatoramaMesh]`, which is a second source of truth for the same fact and
    /// goes stale exactly when a mesh is re-registered.
    ///
    /// The returned object is the renderer's own entry, not a copy: treat its buffers as
    /// read-only, and note a `.private`-storage mesh has no CPU-visible vertices at all
    /// (`objectTriangleSoup()` is empty for it).
    public func registeredMesh(_ kind: MeshKind) -> IlluminatoramaMesh? { meshes[kind] }

    /// Called by `IlluminatoramaMeshHandle.deinit` to evict the entry
    /// when the host drops its last strong reference. Internal because
    /// hosts should not call this directly — letting the handle's
    /// lifetime drive cleanup avoids accidental dangling MeshKind keys
    /// while an SCNNode still references the geometry.
    internal func removeMesh(_ kind: MeshKind) {
        meshes.removeValue(forKey: kind)
        fxs.gpuRepackTasks.removeAll { $0.kind == kind }
    }

    /// **An environment read that is cheap enough for a frame.** `ProcessInfo.processInfo
    /// .environment` builds the WHOLE environment into a new Dictionary on every access. Read per
    /// frame, or per material, that is real time: Daydream Home's material resolver did it once per
    /// resolution and it cost ~1 ms/frame on its Hero fixture. `getenv` walks `environ` without
    /// allocating, and it still sees a runtime `setenv`, which is why these switches are read per
    /// frame rather than cached in a `static let`.
    nonisolated static func frameEnv(_ name: String) -> String? {
        getenv(name).map { String(cString: $0) }
    }

    /// "object-fit: cover" UV sub-rect for presenting the FIXED output canvas
    /// (`outputWidth × outputHeight`) into a display viewport of a possibly
    /// different aspect. Returns `(scaleX, scaleY, offsetX, offsetY)` such that
    /// `texCoord = uv * scale + offset` samples a centered crop of the canvas:
    /// the canvas is scaled to FILL the viewport and the overflow on the longer
    /// axis is cropped, centered — no stretch, no letterbox. Returns identity
    /// `(1, 1, 0, 0)` when the aspects already match (the steady state, so the
    /// present is a 1:1 blit and costs nothing). Build a SceneKit
    /// `background.contentsTransform` from the result; see the host controllers.
    ///
    /// The canvas should be ≥ the viewport on the cropped axis or this upscales
    /// (accepted cost of a fixed canvas — see the controller's tick docstring).
    /// Reallocate render targets at a new size. Targets are dropped first so
    /// the device doesn't hold two copies at the peak; if allocation fails we
    /// log and keep the old size — better a stale image than a crash.
    /// Resize the renderer to a new OUTPUT size. The internal pipeline is
    /// sized to `outputW × internalRenderScale`; pass the same dimensions
    /// you would have passed when there was no SSAA (the SCNView's
    /// drawable size). No-op when neither output nor internal size needs
    /// to change.
    ///
    /// `outputTexture` is preserved across resizes when the OUTPUT
    /// dimensions don't change — this matters for consumers that bound
    /// the texture into a SceneKit material at init time. An SSAA-only
    /// change (internal scale slider) keeps `outputTexture`'s identity
    /// stable so the host's `scene.background.contents = renderer.outputTexture`
    /// binding keeps working without any per-frame rebind. Hosts that
    /// also handle viewport resizes (where output dims DO change) still
    /// need to rebind — the `IlluminatoramaOverlay` does this in its tick
    /// by ObjectIdentifier-comparing the texture before and after.
    /// UV transform (scaleU, scaleV, offsetU, offsetV) that COVER-crops the fixed
    /// render canvas (aspect `outputWidth:outputHeight`) to a live viewport of
    /// `viewportWidth:viewportHeight`, centred — fills the viewport, never
    /// stretches, never letterboxes. Bind it to `background.contentsTransform`
    /// (`m11=scaleU, m22=scaleV, m41=offsetU, m42=offsetV`). This avoids resizing
    /// the renderer to the drawable (which reallocates every GPU target and
    /// resets temporal accumulation, flashing noise on every window drag).
    public func coverUVRect(viewportWidth: Int, viewportHeight: Int) -> SIMD4<Float> {
        guard viewportWidth > 0, viewportHeight > 0,
              outputWidth > 0, outputHeight > 0 else { return SIMD4(1, 1, 0, 0) }
        let canvasAR = Float(outputWidth) / Float(outputHeight)
        let viewAR   = Float(viewportWidth) / Float(viewportHeight)
        if viewAR > canvasAR {
            // Viewport wider than canvas → crop top/bottom (sample full width).
            let scaleV = canvasAR / viewAR
            return SIMD4(1, scaleV, 0, (1 - scaleV) * 0.5)
        } else {
            // Viewport taller/narrower → crop sides (sample full height).
            let scaleU = viewAR / canvasAR
            return SIMD4(scaleU, 1, (1 - scaleU) * 0.5, 0)
        }
    }

    /// Resize to a new OUTPUT size, reallocating every internal target at
    /// `output × internalRenderScale`. Also the path an `internalRenderScale`
    /// change takes (its `didSet` calls this at the unchanged output size), so
    /// the two are one mechanism, not two.
    ///
    /// **Cost — the numbers a host needs before it wires this to an
    /// interaction.** MEASURED on an M1 Max, allocating the exact target list
    /// `makeTargets` builds, output 2560×1440:
    ///
    /// | scale | internal | target set |
    /// |---|---|---|
    /// | 0.5 | 1280×720 | 179 MB |
    /// | 1.0 | 2560×1440 | 715 MB |
    /// | 1.5 | 3840×2160 | 1609 MB |
    ///
    /// (a flat 194 bytes per internal pixel: twenty `rgba16Float` full-res
    /// buffers plus depth ×2, velocity, layer and the SVGF counters).
    ///
    ///   • **CPU side of the swap: ~0.15 ms** (p50 0.14–0.17 ms over n=21
    ///     alternating transitions, any tier pair). Negligible.
    ///   • **First GPU use of the new set: ~30 µs per MB** — Metal backs
    ///     `.private` textures lazily, so the page-in lands on the frame after
    ///     the resize: +5 ms at scale 0.5, +24 ms at 1.0, +44 ms at 1.5, one
    ///     time. THIS is the hitch, not the allocation.
    ///   • **Every temporal accumulator is reset** (TAA, SSAO, SSR and RT-GI
    ///     reproject against nothing next frame), so the image also visibly
    ///     re-converges over the following handful of frames.
    ///
    /// Consequence: a quality-tier flip is a DEBOUNCED, edge-triggered action —
    /// drop on interaction-start, restore once the interaction has settled. Two
    /// transitions per gesture cost roughly 24 ms + 44 ms of page-in against a
    /// saving of ~8.7 ms per frame (1.5 → 1.0 at this canvas size), so it pays
    /// for itself after ~8 dragged frames and is a large net loss if driven per
    /// frame. (Keeping two target sets alive to swap between them was considered
    /// and rejected: 2.3 GB resident to avoid a transition that happens twice
    /// per gesture. Rendering into a sub-rect of max-size targets — viewport
    /// scissor — was also rejected: it removes the realloc but needs a "virtual
    /// size" threaded through every normalised-UV site in the shader library
    /// (tonemap magnify, TAA/SSR reprojection, bloom chain), risks half-texel
    /// and edge-bleed errors at every one of them, and keeps the full 1.6 GB
    /// resident even in draft mode.)
    public func resize(width: Int, height: Int) {
        // Same ceiling as `init` — a host that asks for more than `canvasBudget`
        // allows (a canvas-scale slider, an SSAA write) gets the largest canvas
        // it is allowed to have, at the requested aspect.
        let budgeted = Self.budgeted(outputW: width, outputH: height,
                                     scale: internalRenderScale)
        let outW = budgeted.w
        let outH = budgeted.h
        if budgeted.scale != internalRenderScale { internalRenderScale = budgeted.scale }
        let (inW, inH) = Self.internalDims(outputW: outW, outputH: outH,
                                            scale: budgeted.scale)
        guard outW != self.outputWidth || outH != self.outputHeight ||
              inW  != self.width       || inH  != self.height else { return }
        let outputDimsChanged = (outW != self.outputWidth || outH != self.outputHeight)
        do {
            let t = try Self.makeTargets(device: device,
                                          internalW: inW, internalH: inH,
                                          outputW: outW, outputH: outH)
            self.gbs.gbufferAlbedoMet    = t.albedoMet
            self.gbs.gbufferNormalRgh    = t.normalRgh
            self.gbs.gbufferEmission     = t.emission
            self.gbs.depthTexture        = t.depth
            self.gbs.previousDepthTexture = try Self.makeDepthLike(device: device, t.depth)
            self.gbs.hdrTexture          = t.hdr
            self.aos.aoTexture           = t.ao
            self.pfx.hdrCompositeTexture = t.hdrComposite
            self.rts.rtDiffuseTexture    = t.rtDiffuse
            self.pfx.velocityTexture     = t.velocity
            self.gbs.gbufferLayer        = t.layer
            self.gbs.gbufferMaterial     = nil          // DH-0140 — re-made at the new size on first use
            self.pfx.historyA            = t.historyA
            self.pfx.historyB            = t.historyB
            self.pfx.taaResolvedTexture  = t.taaResolved
            self.pfx.bloomDownChain      = t.bloomDown
            self.pfx.bloomUpChain        = t.bloomUp
            self.pfx.halationBright = t.halationBright
            self.pfx.halationBlurH  = t.halationBlurH
            self.pfx.halationBlurV  = t.halationBlurV
            self.aos.aoFilteredTexture   = t.aoFiltered
            self.aos.aoHistoryA          = t.aoHistoryA
            self.aos.aoHistoryB          = t.aoHistoryB
            self.ssrs.ssrRawTexture       = t.ssrRaw
            self.ssrs.ssrHistoryA         = t.ssrHistoryA
            self.ssrs.ssrHistoryB         = t.ssrHistoryB
            self.rts.rtGIHistoryA        = t.rtGIHistoryA
            self.rts.rtGIHistoryB        = t.rtGIHistoryB
            self.aos.aoSampleCount       = t.aoSampleCount
            self.ssrs.ssrSampleCount      = t.ssrSampleCount
            self.rts.giSampleCount       = t.giSampleCount
            self.rts.giVariance          = t.giVariance
            self.rts.giVarianceB         = t.giVarianceB
            self.rts.giAtrousA           = t.giAtrousA
            self.rts.giAtrousB           = t.giAtrousB
            self.ssrs.sssDiffuseTexture   = t.sssDiffuse
            self.ssrs.sssBlurTmpTexture   = t.sssBlurTmp
            self.ssrs.sssBlurOutTexture   = t.sssBlurOut
            // Preserve outputTexture identity for SSAA-only changes — see
            // the docstring. On an output-size change, rebuild the whole
            // presented-buffer pool at the new size (reusing `t.ldr` as
            // slot 0) and reset the present bookkeeping. `t.ldr` would
            // otherwise go out of scope and be freed with the local `t`.
            if outputDimsChanged {
                var pool: [MTLTexture] = [t.ldr]
                for i in 1..<Self.ldrPoolCount {
                    if let extra = Self.makeLDRTexture(
                        device: device, w: outW, h: outH,
                        label: "Illuminatorama.ldr.\(i)") {
                        pool.append(extra)
                    }
                }
                Self.clearLDRTextures(pool, on: commandQueue)
                self.pfx.ldrPool = pool
                // KEEP PRESENTING THE LAST FINISHED FRAME until one at the new size completes.
                // `outputTexture` used to become `t.ldr` here — a texture no frame has written —
                // and the very next `present(to:)` blitted it: a fresh private texture reads back
                // as solid 255,0,255, so every live-resize step flashed fuchsia. The old texture is
                // fully written and still retained here; `present` samples it in normalised UVs, so
                // for the frame or two until the new size lands it is merely stretched. No pool
                // slot is "presented" yet (-1), so every new slot is writable and the first one to
                // complete is promoted.
                self.pfx.presentedIdx = -1
                self.pfx.prevPresentedIdx = -1
                self.pfx.presentSync.reset()
                self.pfx.tonemapWriteTarget = t.ldr
            }
            self.width        = inW
            self.height       = inH
            self.outputWidth  = outW
            self.outputHeight = outH
            // Internal and output share the same multiplier, so aspect is
            // identical for both. Drive camera off the output size as the
            // canonical user-facing dimension.
            self.camera.aspect = Float(outW) / Float(outH)
            // Resize destroys all history textures — next frame has nothing
            // valid to reproject against for any temporal pass.
            self.pfx.taaNeedsFirstFrame  = true
            self.aos.aoNeedsFirstFrame   = true
            self.ssrs.ssrNeedsFirstFrame  = true
            self.rtGINeedsFirstFrame = true
        } catch {
            Self.log.error("Illuminatorama resize(out \(outW)x\(outH) / int \(inW)x\(inH)) failed: \(error.localizedDescription) — keeping out \(self.outputWidth)x\(self.outputHeight) int \(self.width)x\(self.height)")
        }
    }

    /// Map an output size + scale to the internal render-target size,
    /// rounding to multiples of 2 so the half-res bloom / AO textures
    /// don't end up off-by-one on odd output dimensions.
    ///
    /// The clamp used to be `max(1.0, scale)` — sub-native rendering was
    /// impossible, which meant the ONLY direction the pixel cost could move was
    /// up. A deferred frame is fill-bound, so that clamp was the thing standing
    /// between the app and a draft/navigation quality tier. It is now
    /// `internalRenderScaleRange` (0.5 … 4.0); see `minInternalRenderScale`.
    ///
    /// Rounding notes: `& ~1` truncates to an even number AFTER `.rounded()`, so
    /// an odd product rounds DOWN by one — deliberate, because every derived
    /// chain divides (÷2 for SSAO/bloom/halation) and an even parent is
    /// what keeps those aligned. `nonZeroSize` never returns 0: a 0-wide texture
    /// is a hard Metal allocation failure, and `resize` would then silently keep
    /// the old size forever.
    ///
    /// `internal` (not `private`) so the pure-arithmetic contract is testable
    /// without a Metal device — `swift test` on this package has no metallib and
    /// cannot construct a renderer at all.
    nonisolated static func internalDims(outputW: Int, outputH: Int,
                                         scale: Float) -> (Int, Int) {
        let s = min(maxInternalRenderScale, max(minInternalRenderScale, scale))
        func nonZeroSize(_ v: Int) -> Int {
            max(2, Int((Float(v) * s).rounded()) & ~1)
        }
        return (nonZeroSize(max(1, outputW)), nonZeroSize(max(1, outputH)))
    }

    /// Real frame-delivery jitter probe (issue #65). Records the interval since
    /// the last `outputTexture` swap and writes a rolling p50/p95/p99 to
    /// `VIZ_ILLUMI_SWAP_PATH`. Main-actor only (called from promoteCompletedBuffer),
    /// so no locking. No-op cost when the probe env is off (guarded at the call).
    func recordSwapInterval() {
        let now = CACurrentMediaTime()
        defer { gbs.swapLastTime = now }
        guard gbs.swapLastTime > 0 else { return }   // first swap: no interval yet
        let ms = (now - gbs.swapLastTime) * 1000.0
        gbs.swapIntervalsMs.append(ms)
        if gbs.swapIntervalsMs.count > 4000 { gbs.swapIntervalsMs.removeFirst(gbs.swapIntervalsMs.count - 4000) }
        guard let p = gbs.swapProbePath, gbs.swapIntervalsMs.count >= 2 else { return }
        let sorted = gbs.swapIntervalsMs.sorted()
        func pct(_ q: Double) -> Double { sorted[min(sorted.count - 1, max(0, Int(q * Double(sorted.count - 1)))) ] }
        let mean = sorted.reduce(0, +) / Double(sorted.count)
        let p50 = pct(0.50), p95 = pct(0.95), p99 = pct(0.99)
        let fps = mean > 0 ? 1000.0 / mean : 0
        let line = String(format: "swaps=%d fps=%.1f meanMs=%.2f p50=%.2f p95=%.2f p99=%.2f maxMs=%.2f\n",
                          sorted.count + 1, fps, mean, p50, p95, p99, sorted.last ?? 0)
        if let d = line.data(using: .utf8) { try? d.write(to: URL(fileURLWithPath: p)) }
    }

    func encodeGBufferPass(_ cb: MTLCommandBuffer) {
        // DH-0140 — the lane was decided when this frame's uniforms were written; pick the
        // matching pipelines once. The extended variants exist whenever `extendedGBufferActive`.
        let extended = extendedGBufferActive
        let basePipe: MTLRenderPipelineState = (extended ? gbs.gbufferPipelineExt : nil) ?? gbs.gbufferPipeline
        let prevPipe: MTLRenderPipelineState = (extended ? gbs.gbufferPipelinePrevVertsExt : nil) ?? gbs.gbufferPipelinePrevVerts
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = gbs.gbufferAlbedoMet
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[1].texture = gbs.gbufferNormalRgh
        pass.colorAttachments[1].loadAction = .clear
        pass.colorAttachments[1].storeAction = .store
        pass.colorAttachments[1].clearColor = MTLClearColor(red: 0.5, green: 0.5, blue: 0, alpha: 0)
        pass.colorAttachments[2].texture = gbs.gbufferEmission
        pass.colorAttachments[2].loadAction = .clear
        pass.colorAttachments[2].storeAction = .store
        pass.colorAttachments[2].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        // Phase 2.7 — velocity. Sky pixels (no geometry) stay at zero, which
        // means the TAA resolve treats them as stationary — fine for static
        // skies and visibly slightly smeary for moving clouds.
        pass.colorAttachments[3].texture = pfx.velocityTexture
        pass.colorAttachments[3].loadAction = .clear
        pass.colorAttachments[3].storeAction = .store
        pass.colorAttachments[3].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        // Light-layer target. Cleared to the ALL-BITS default (0xFFFFFFFF) so every
        // sky / no-geometry pixel — and every instance that never sets a layer — is
        // lit by every light exactly as before. The R32Uint clear reads its integer
        // from clearColor.red.
        pass.colorAttachments[4].texture = gbs.gbufferLayer
        pass.colorAttachments[4].loadAction = .clear
        pass.colorAttachments[4].storeAction = .store
        pass.colorAttachments[4].clearColor = MTLClearColor(red: Double(UInt32.max), green: 0, blue: 0, alpha: 0)
        if extended, let material = gbs.gbufferMaterial {
            // Clear to 0 = "not written": impostors and any legacy pipeline that draws into this
            // pass without declaring color(5) leave the clear value, and every reader falls back.
            pass.colorAttachments[5].texture = material
            pass.colorAttachments[5].loadAction = .clear
            pass.colorAttachments[5].storeAction = .store
            pass.colorAttachments[5].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        }
        pass.depthAttachment.texture = gbs.depthTexture
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1.0

        guard let enc = timedRenderEncoder(cb, pass, "gbuffer") else { return }
        enc.label = "Illuminatorama.gbuffer"
        enc.setRenderPipelineState(basePipe)
        enc.setDepthStencilState(gbs.depthState)
        enc.setFrontFacing(.counterClockwise)
        // Cull mode is a PER-MESH decision here (a two-sided mesh draws `.none`),
        // so it is set inside the draw loop below and deliberately NOT set
        // pass-wide: a pass-wide `.back` is immediately re-issued by the first
        // group, and re-issuing the same value per group is exactly what Metal's
        // validation layer reports as `redundant setCullMode.` /
        // `previous setCullMode was unused.`. Those are severity-2 warnings, but
        // under the debug layer's assert warning mode (Xcode's GPU frame capture
        // / break-on-validation) `_MTLMessageContextEnd…OrAbort` takes the ABORT
        // arm — so a redundant state set is a crash, not a log line. Track the
        // last value and set only on a change, exactly as the shadow passes do.
        // Metal's own default is `.none`, and every drawn group states its cull
        // explicitly, so dropping the pass-wide set changes no pixels.
        var cull: MTLCullMode? = nil

        // Phase 4.0/4.1 — bind the per-material texture atlases once for the
        // whole G-buffer pass. Per-instance slice indices select the slice
        // within each atlas (`< 0` → no sampling, fall back to the per-
        // instance scalar/colour). albedoAtlas is sRGB; nonColorAtlas is
        // linear so metallic/roughness round-trip exactly.
        enc.setFragmentTexture(albedoAtlas.texture, index: 0)
        enc.setFragmentTexture(nonColorAtlas.texture, index: 1)
        // Phase 4.25 (issue #60 item 5) — per-slice UV-scale tables that let the
        // shader tile letterboxed (aspect-preserved) non-square textures at
        // their true aspect. `float2` per slice; indexed by the instance's slice
        // id. Fragment buffer(2) is the instance buffer (set per group below),
        // (4) is unused by the fragment stage — these live at (3)/(5) to match
        // `illumi_fs`'s signature.
        enc.setFragmentBuffer(albedoAtlas.uvScaleBuffer, offset: 0, index: 3)
        enc.setFragmentBuffer(nonColorAtlas.uvScaleBuffer, offset: 0, index: 5)
        // S2.5 — per-slice mean tables (μ for the variance-preserving hex blend). Pass-wide
        // like the UV-scale tables; fragment (7)/(8) are otherwise unused by this pass, and
        // (6) is the superquadric param buffer set per-group below.
        enc.setFragmentBuffer(albedoAtlas.sliceMeanBuffer, offset: 0, index: 7)
        enc.setFragmentBuffer(nonColorAtlas.sliceMeanBuffer, offset: 0, index: 8)
        // Phase 7 — bind the per-frame uniforms to the FRAGMENT stage too (they were
        // already bound to vertex at buffer(1)) so `illumi_fs` can read
        // `antiTilingStrength`. Pass-wide (constant across the draw loop).
        enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 1)
        // Phase 4.12 — instanced draws via the recipe built in
        // `uploadInstances`. One draw call per mesh kind, with the
        // current/previous instance buffer offsets pointed at the start
        // of that group's contiguous slice. The vertex shader's
        // `instances[iid]` indexing then walks the slice naturally.
        // Killed the prior per-instance inner loop — FloatingFlowers+
        // drops from ~1500 draw calls per frame to ~10.
        let instStride = MemoryLayout<IlluminatoramaInstance>.stride
        // Issue #65 — map each deforming GPU-mesh kind to its prev-position side
        // buffer so those groups can draw with the kUsePrevVerts pipeline. Built
        // once per pass (gpuRepackTasks is tiny — one entry per registered mesh).
        var prevPosByKind: [MeshKind: MTLBuffer] = [:]
        for task in fxs.gpuRepackTasks { prevPosByKind[task.kind] = task.prevPositionBuffer }
        // Track which pipeline is bound to avoid redundant state switches when the
        // groups are all ordinary meshes (the common case).
        var prevVertsBound = false
        var interiorBound = false
        var substrateBiased = false
        let cullClip = cullVolume(gbs.gbufferCullVP)   // DH-0534
        for (gi, group) in gbs.meshGroups.enumerated() {
            // Superquadric impostor box kinds are drawn by the impostor pipeline
            // below; RT-proxy kinds exist only in the TLAS; shadow-only kinds are
            // lighting occluders that draw solely into the shadow maps. All skipped here.
            if gbs.impostorMeshKinds.contains(group.kind) ||
               rts.rtProxyMeshKinds.contains(group.kind) ||
               shadowOnlyMeshKinds.contains(group.kind) { continue }
            guard let mesh = meshes[group.kind] else { continue }
            if shouldCull(gi, group, mesh, cullClip, drawn: \.gbufferDrawn, culled: \.gbufferCulled) { continue }
            // Two-sided meshes (open / dynamic MC fluid surfaces) render cull
            // `.none` so they don't go hollow when their back side faces the
            // camera; the fragment shader flips the normal for back faces.
            let want: MTLCullMode = mesh.doubleSided ? .none : .back
            if cull != want { cull = want; enc.setCullMode(want) }
            // The substrate takes a depth-buffer bias so a millimetre of lift on top of it keeps
            // winning at any distance (see `substrateMeshKinds`). Tracked like `cull` — one
            // state change per run of groups, not per group.
            let wantBias = substrateMeshKinds.contains(group.kind)
            if wantBias != substrateBiased {
                substrateBiased = wantBias
                enc.setDepthBias(wantBias ? substrateDepthBias : 0, slopeScale: 0, clamp: 0)
            }
            let off = instStride * group.start
            // Issue #65 — deforming GPU meshes draw with the kUsePrevVerts pipeline
            // and bind their per-vertex prev positions at buffer(5); every other
            // group uses the base pipeline (which declares no buffer(5)).
            let prevPos = prevPosByKind[group.kind]
            // Interior-mapped kinds (opt-in): their own pipeline + params at fragment buffer(9).
            if let imap = interiorMappedMeshKinds[group.kind], prevPos == nil,
               let ip = interiorGBufferPipeline(extended: extended) {
                if !interiorBound { enc.setRenderPipelineState(ip) }
                interiorBound = true
                prevVertsBound = false
                var gpu = imap.gpu
                enc.setFragmentBytes(&gpu, length: MemoryLayout<IlluminatoramaInteriorMapping.GPU>.stride, index: 9)
            } else if prevPos != nil {
                if !prevVertsBound || interiorBound { enc.setRenderPipelineState(prevPipe) }
                prevVertsBound = true
                interiorBound = false
            } else if prevVertsBound || interiorBound {
                enc.setRenderPipelineState(basePipe)
                prevVertsBound = false
                interiorBound = false
            }
            enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
            enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
            enc.setFragmentBuffer(currentInstanceBuffer, offset: off, index: 2)
            // Phase 2.7 — previous-frame instances at buffer(4) for the
            // motion-vector reconstruction. Same per-group offset as
            // current — the ping-pong buffers share the grouped layout
            // because the host's `instances` order is stable across
            // frames for a static scene topology.
            enc.setVertexBuffer(previousInstanceBuffer, offset: off, index: 4)
            // Issue #65 — per-vertex previous positions for the deforming-mesh
            // motion vector. Only the kUsePrevVerts pipeline reads buffer(5).
            if let prevPos { enc.setVertexBuffer(prevPos, offset: 0, index: 5) }
            enc.drawIndexedPrimitives(
                type: .triangle,
                indexCount: mesh.indexCount,
                indexType: mesh.indexType,
                indexBuffer: mesh.indexBuffer,
                indexBufferOffset: 0,
                instanceCount: group.count,
                baseVertex: 0,
                baseInstance: 0
            )
        }

        // ── Perfect analytic superquadric impostors ───────────────────────────
        // Drawn into the SAME G-buffer pass with the impostor pipeline: each
        // instance's [-1,1] bounding-box mesh is rasterized (BACK faces — cull
        // `.front` — so the primitive survives the camera entering the box), and
        // the fragment ray-traces the analytic surface, writing the G-buffer +
        // analytic depth + analytic motion vectors. The param buffer is bound at
        // the same per-group offset as the instance buffer (grouped-order aligned).
        if substrateBiased { substrateBiased = false; enc.setDepthBias(0, slopeScale: 0, clamp: 0) }

        if let sqPipe = gbs.superquadricImpostorPipeline, !gbs.impostorMeshKinds.isEmpty {
            enc.setRenderPipelineState(sqPipe)
            if cull != .front { cull = .front; enc.setCullMode(.front) }
            let pStride = MemoryLayout<IlluminatoramaInstance.SuperquadricParam>.stride
            for group in gbs.meshGroups where gbs.impostorMeshKinds.contains(group.kind) {
                guard let mesh = meshes[group.kind] else { continue }
                let off  = instStride * group.start
                let poff = pStride * group.start
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
                enc.setVertexBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 1)
                enc.setFragmentBuffer(currentInstanceBuffer, offset: off, index: 2)
                enc.setFragmentBuffer(previousInstanceBuffer, offset: off, index: 4)
                enc.setFragmentBuffer(superquadricParamBuffer, offset: poff, index: 6)
                enc.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: mesh.indexCount,
                    indexType: mesh.indexType,
                    indexBuffer: mesh.indexBuffer,
                    indexBufferOffset: 0,
                    instanceCount: group.count,
                    baseVertex: 0,
                    baseInstance: 0
                )
            }
        }
        enc.endEncoding()
    }

    func dispatch(_ enc: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          width: Int, height: Int,
                          maxThreadsOverride: Int? = nil) {
        let tgw = pipeline.threadExecutionWidth
        // Default = the full threadgroup (max occupancy by thread count). For
        // register-heavy kernels (RT) a SMALLER group can run faster — more
        // threadgroups resident per core. `maxThreadsOverride` lets a specific
        // pass cap the total threads (#60 task 4 occupancy tuning).
        let cap = pipeline.maxTotalThreadsPerThreadgroup
        let total = max(tgw, min(cap, maxThreadsOverride ?? cap))
        let tgh = max(1, total / tgw)
        let threadgroup = MTLSize(width: tgw, height: tgh, depth: 1)
        let groups = MTLSize(
            width:  (width  + tgw - 1) / tgw,
            height: (height + tgh - 1) / tgh,
            depth: 1
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroup)
    }

    /// 3D dispatch covering (faceSize × faceSize × 6) for cubemap-baking
    /// kernels that take `[[thread_position_in_grid]]` as `uint3` and use
    /// `gid.z` as the face index. Threadgroups are 8×8×1 — depth=1 because
    /// faces don't share state and that keeps the threadgroup tile small
    /// enough to fit on every device.
    func dispatchCubeFaces(_ enc: MTLComputeCommandEncoder,
                                   pipeline: MTLComputePipelineState,
                                   faceSize: Int) {
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let groups = MTLSize(
            width:  (faceSize + tg.width - 1)  / tg.width,
            height: (faceSize + tg.height - 1) / tg.height,
            depth:  6
        )
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }

    // ── Upload helpers ────────────────────────────────────────────────────────

    func ensureBufferCapacity() {
        if instances.count > gbs.instanceCapacity {
            let newCap = max(instances.count, gbs.instanceCapacity * 2)
            let stride = MemoryLayout<IlluminatoramaInstance>.stride * newCap
            // Grow BOTH ping-pong buffers in lockstep — the previous-frame
            // buffer is now uninitialized for motion-vector purposes, so we
            // also force TAA back into first-frame mode.
            if let a = device.makeBuffer(length: stride, options: .storageModeShared),
               let b = device.makeBuffer(length: stride, options: .storageModeShared) {
                a.label = "Illuminatorama.instancesA"
                b.label = "Illuminatorama.instancesB"
                gbs.instanceBufferA = a
                gbs.instanceBufferB = b
                gbs.instanceCapacity = newCap
                pfx.taaNeedsFirstFrame = true
            }
        }
        // Grow-on-demand for the ring-buffered per-frame inputs. Reallocate the
        // WHOLE ring (all maxFramesInFlight slots) so every in-flight frame keeps
        // a same-size buffer — growing only the current slot would let an
        // in-flight frame bind the old (too-small) slot next time round the ring.
        // Slots are uploaded fresh each frame, so there is nothing to copy over.
        func growRing<T>(_ ring: inout [MTLBuffer], _ capacity: inout Int,
                         needed: Int, type: T.Type, label: String) {
            guard needed > capacity else { return }
            let newCap = max(needed, capacity * 2)
            var nr: [MTLBuffer] = []
            nr.reserveCapacity(ring.count)
            for i in 0..<ring.count {
                guard let nb = device.makeBuffer(
                    length: MemoryLayout<T>.stride * newCap, options: .storageModeShared
                ) else { return }   // alloc failed — keep the old ring, retry next frame
                nb.label = "\(label).\(i)"
                nr.append(nb)
            }
            ring = nr
            capacity = newCap
        }
        // Keep the parallel superquadric param ring at least as large as the
        // instance buffer (indexed by the same grouped instance id).
        growRing(&gbs.superquadricParamRing, &gbs.superquadricParamCapacity,
                 needed: instances.count, type: IlluminatoramaInstance.SuperquadricParam.self,
                 label: "Illuminatorama.superquadricParams")
        growRing(&shd.pointLightRing, &shd.pointLightCapacity,
                 needed: pointLights.count, type: IlluminatoramaPointLight.self,
                 label: "Illuminatorama.pointLights")
        growRing(&shd.spotLightRing, &shd.spotLightCapacity,
                 needed: spotLights.count, type: IlluminatoramaSpotLight.self,
                 label: "Illuminatorama.spotLights")
        growRing(&lts.areaLightRing, &lts.areaLightCapacity,
                 needed: areaLights.count, type: IlluminatoramaAreaLight.self,
                 label: "Illuminatorama.areaLights")
        growRing(&lts.extraDirectionalRing, &lts.extraDirectionalCapacity,
                 needed: extraDirectionals.count, type: IlluminatoramaDirectionalLight.self,
                 label: "Illuminatorama.extraDirectionals")
        growRing(&lts.interiorApertureRing, &lts.interiorApertureCapacity,
                 needed: interiorApertures.count, type: IlluminatoramaInteriorAperture.self,
                 label: "Illuminatorama.interiorApertures")
    }

    func uploadInstances() {
        // PERF (static-scene skip): at ≥2 stable frames BOTH ping-pong buffers
        // already hold exactly this content (each change is written twice, once
        // per buffer), `meshGroups` is the grouping of that same content, and
        // prev==current means zero motion vectors — which is precisely correct
        // for content that didn't move. Skipping avoids the per-frame regroup
        // dictionary, the full buffer rewrite, and the superquadric inverse
        // recomputes. Capacity growth can't be missed: growth implies the count
        // changed, which resets `instanceStableFrames`.
        if gbs.instanceStableFrames >= 2 { staticSkipStats.uploadsSkipped += 1; return }
        // After the ping-pong toggle at the top of render(), the buffer that
        // WAS previous last frame is now "current" — we overwrite it with
        // this frame's data. The other buffer (now "previous") still holds
        // last frame's data, which the G-buffer vertex shader reads for
        // motion-vector reconstruction.
        //
        // Phase 4.12 — write in GROUPED order so all instances of the same
        // `MeshKind` land in a contiguous run. That lets every pass that
        // iterates the scene's geometry (G-buffer, cascaded + spot
        // shadows) replace its per-instance loop with one
        // `drawIndexedPrimitives(..., instanceCount: group.count)` per
        // mesh kind. FloatingFlowers+ goes from ~1500 draw calls to ~10
        // without any shader changes — the vertex shader's
        // `instances[iid]` lookup walks the contiguous slice naturally
        // because we set the vertex/fragment buffer offsets per group.
        //
        // Grouping stability across frames: the host (`extractor`) walks
        // the scene graph in DFS order, so the same scene topology
        // produces the same first-seen ordering of mesh kinds + the same
        // intra-group instance order. That's what motion-vector TAA
        // relies on — `prevInstances[iid]` reads the same logical
        // instance as `instances[iid]` because both buffers used the
        // same grouping last → this frame.
        gbs.meshGroups.removeAll(keepingCapacity: true)
        gbs.meshGroupRange.removeAll(keepingCapacity: true)
        gbs.meshGroupsVersion &+= 1   // DH-0534: the cull bounds describe the old grouping now
        guard !instances.isEmpty else { return }

        // VEGETATION WIND, LIVENESS. `treeWindStrength` is a global, but as of 2026-08-17 the
        // displacement also needs a per-DRAW `windScale` (see the property's own note: without
        // it the gust reached any mesh whose tangents pointed along the sway axis, because a
        // tangent is a TBN basis and not a sway weight). A host that raises the strength and
        // never marks its foliage instances therefore gets NO wind and no error — the inert
        // feature this renderer keeps re-learning. Say so once, on the frame it happens.
        if treeWindStrength > 0, !pfx.warnedWindWithoutOptIn,
           !instances.contains(where: { $0.data.windScale > 0 }) {
            pfx.warnedWindWithoutOptIn = true
            print("[illuminatorama] treeWindStrength=\(treeWindStrength) but no instance sets "
                  + "windScale>0 — the vegetation wind is INERT. Set `windScale = 1` on the "
                  + "foliage draws whose vertices pack (swayWeight, phase, flutter) in tangent.")
        }

        // First pass — bucket by mesh kind, preserving first-seen order
        // so a stable scene gives a stable buffer layout.
        var byKind: [MeshKind: [Int]] = [:]
        byKind.reserveCapacity(8)
        var kindOrder: [MeshKind] = []
        kindOrder.reserveCapacity(8)
        for (srcIdx, ref) in instances.enumerated() {
            if byKind[ref.meshKind] == nil {
                byKind[ref.meshKind] = []
                kindOrder.append(ref.meshKind)
            }
            byKind[ref.meshKind, default: []].append(srcIdx)
        }

        // Second pass — write into the buffer in grouped order, building
        // the per-group draw recipe alongside.
        let dst = currentInstanceBuffer.contents().bindMemory(
            to: IlluminatoramaInstance.self, capacity: instances.count
        )
        // Parallel superquadric param buffer, written in the SAME grouped order so
        // `params[iid]` lines up with `instances[iid]` in the impostor draw. Most
        // entries are unused (only impostor instances carry a shape); cheap to fill.
        let sqDst = superquadricParamBuffer.contents().bindMemory(
            to: IlluminatoramaInstance.SuperquadricParam.self, capacity: instances.count
        )
        var dstIdx = 0
        for kind in kindOrder {
            guard let srcIndices = byKind[kind] else { continue }
            let groupStart = dstIdx
            for srcIdx in srcIndices {
                let ref = instances[srcIdx]
                dst[dstIdx] = ref.data
                if let s = ref.superquadricShape {
                    let isEllipsoid = (abs(s.x - 1) < 1e-3 && abs(s.y - 1) < 1e-3)
                    sqDst[dstIdx] = IlluminatoramaInstance.SuperquadricParam(
                        invModel: simd_inverse(ref.data.modelMatrix),
                        shape: SIMD4(s.x, s.y, isEllipsoid ? 1 : 0, 0))
                } else {
                    sqDst[dstIdx] = IlluminatoramaInstance.SuperquadricParam(
                        invModel: matrix_identity_float4x4, shape: .zero)
                }
                dstIdx += 1
            }
            gbs.meshGroups.append(MeshDrawGroup(
                kind: kind, start: groupStart, count: srcIndices.count))
            gbs.meshGroupRange[kind] = groupStart ..< (groupStart + srcIndices.count)
        }
    }

    /// DH-0534 — rebuild `meshGroupCullBounds` when its inputs changed. Everything the CPU cannot
    /// see is UNBOUNDED and always draws: a host `onEncodeGPUInstances` kernel may overwrite ANY
    /// instance slot (so with the hook set, nothing is culled at all); `gpuRepackTasks` rewrite a
    /// kind's vertices (which also covers the deforming meshes behind `prevPosByKind`); and a mesh
    /// built on caller-owned buffers has no CPU-visible vertices (`cpuAuthoredVertices == false`).
    func computeCullBounds() {
        let hostWritesInstances = onEncodeGPUInstances != nil
        let key = CullBoundsKey(groupsVersion: gbs.meshGroupsVersion, wind: treeWindStrength,
                                hostWritesInstances: hostWritesInstances,
                                repackCount: fxs.gpuRepackTasks.count)
        if key == gbs.cullBoundsKey, gbs.meshGroupCullBounds.count == gbs.meshGroups.count { return }
        gbs.cullBoundsKey = key
        gbs.meshGroupCullBounds.removeAll(keepingCapacity: true)
        let total = gbs.meshGroups.last.map { $0.start + $0.count } ?? 0
        guard total > 0 else { return }
        let repacked = Set(fxs.gpuRepackTasks.map(\.kind))
        let inst = currentInstanceBuffer.contents()
            .bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        for group in gbs.meshGroups {
            guard !hostWritesInstances, !repacked.contains(group.kind),
                  let mesh = meshes[group.kind], mesh.cpuAuthoredVertices,
                  let local = mesh.boundingSphere else {
                gbs.meshGroupCullBounds.append(.unbounded)
                continue
            }
            var bounds = IlluminatoramaCullBounds.empty
            var bounded = true
            for i in group.start ..< group.start + group.count {
                guard let sphere = IlluminatoramaCullBounds.worldSphere(
                    local: local, instance: inst[i], treeWindStrength: treeWindStrength,
                    windMax: { mesh.windAttributeMax }) else { bounded = false; break }
                bounds.formUnion(center: sphere.center, radius: sphere.radius)
            }
            gbs.meshGroupCullBounds.append(bounded ? bounds : .unbounded)
        }
    }

    /// The clip volume a pass culls against, or nil when culling is off (nil ⇒ draw everything).
    func cullVolume(_ vp: simd_float4x4) -> IlluminatoramaClipVolume? {
        visibilityCullingEnabled ? IlluminatoramaClipVolume(vp) : nil
    }

    /// DH-0534 — true when group `gi` cannot produce a fragment in a pass rasterising with `clip`.
    /// Counts the group into `drawn` or `culled` either way, so the stats are complete.
    @inline(__always)
    func shouldCull(_ gi: Int, _ group: MeshDrawGroup, _ mesh: IlluminatoramaMesh,
                            _ clip: IlluminatoramaClipVolume?,
                            drawn: WritableKeyPath<IlluminatoramaVisibilityCullStats, Int>,
                            culled: WritableKeyPath<IlluminatoramaVisibilityCullStats, Int>) -> Bool {
        if let clip, gi < gbs.meshGroupCullBounds.count, clip.excludes(gbs.meshGroupCullBounds[gi]) {
            visibilityCullStats[keyPath: culled] += 1
            visibilityCullStats.trianglesCulled += (mesh.indexCount / 3) * group.count
            return true
        }
        visibilityCullStats[keyPath: drawn] += 1
        return false
    }

    // ── Target allocation ─────────────────────────────────────────────────────

    struct Targets {
        var albedoMet: MTLTexture
        var normalRgh: MTLTexture
        var emission: MTLTexture
        var depth: MTLTexture
        var hdr: MTLTexture
        var ao: MTLTexture
        var hdrComposite: MTLTexture
        var rtDiffuse: MTLTexture    // full-res RT diffuse (shadow+GI) pre-denoise
        // Phase 2.7
        var velocity: MTLTexture
        // Light-layer G-buffer target (R32Uint). Carries each fragment's `layer`
        // bitfield so the deferred lighting kernel can mask per-room lights. Cleared
        // to 0xFFFFFFFF (sky/no-geometry) — the all-bits default means no scene that
        // leaves layers/masks untouched sees any change.
        var layer: MTLTexture
        var historyA: MTLTexture
        var historyB: MTLTexture
        /// The TAA resolve's DOWNSTREAM output — the sharpened frame bloom and
        /// tonemap read. Separate from the ping-pong pair because those two are
        /// the ACCUMULATOR, and the acutance sharpen must not compound inside it.
        var taaResolved: MTLTexture
        // Bloom mip pyramid (S1.2) — see `bloomDownChain` for why these are
        // separate per-level textures rather than one mipmapped resource.
        var bloomDown: [MTLTexture]      // level 0 = internal/2, each halving
        var bloomUp: [MTLTexture]        // one SHORTER than `bloomDown`
        // Film halation — half-res (see `halationBright`).
        var halationBright: MTLTexture
        var halationBlurH: MTLTexture
        var halationBlurV: MTLTexture
        var ldr: MTLTexture
        // Phase 4.39 denoiser
        var aoFiltered: MTLTexture   // half-res, after bilateral
        var aoHistoryA: MTLTexture   // half-res r16Float ping-pong
        var aoHistoryB: MTLTexture
        var ssrRaw: MTLTexture       // full-res SSR gather delta
        var ssrHistoryA: MTLTexture  // full-res rgba16Float ping-pong
        var ssrHistoryB: MTLTexture
        var rtGIHistoryA: MTLTexture // full-res rgba16Float ping-pong (RT GI temporal)
        var rtGIHistoryB: MTLTexture
        // SVGF denoiser (Phase 4.44)
        var aoSampleCount: MTLTexture    // half-res r16Float — accumulated frame count per SSAO pixel
        var ssrSampleCount: MTLTexture   // full-res r16Float — accumulated frame count per SSR pixel
        var giSampleCount: MTLTexture    // full-res r16Float — accumulated frame count per RT GI pixel
        var giVariance: MTLTexture       // full-res r16Float — variance ping-pong A
        var giVarianceB: MTLTexture      // full-res r16Float — variance ping-pong B
        var giAtrousA: MTLTexture        // full-res rgba16Float — à-trous ping-pong A
        var giAtrousB: MTLTexture        // full-res rgba16Float — à-trous ping-pong B
        // Issue #65 — screen-space SSS. `sssDiffuse` holds the diffuse-lit term the
        // lighting pass peels off SSS-flagged pixels (rgb) + the mask (a); the blur
        // ping-pongs sssDiffuse → sssBlurTmp (H) → sssBlurOut (V); the composite
        // reads sssDiffuse (sharp) + sssBlurOut (blurred). Only written/read when a
        // scene sets sssStrength > 0; otherwise allocated-but-idle.
        var sssDiffuse: MTLTexture       // full-res rgba16Float — diffuse + mask
        var sssBlurTmp: MTLTexture       // full-res rgba16Float — horizontal-blur intermediate
        var sssBlurOut: MTLTexture       // full-res rgba16Float — separable-blur result
    }

    /// Standalone sibling of the G-buffer depth, sized to match it, used as the
    /// PREVIOUS frame's depth in the TAA resolve's disocclusion test. Not part of
    /// the texture pool — it's blit-filled from `depthTexture` at the end of each
    /// frame. `.shaderRead` so the resolve kernel can `read()` it; `.renderTarget`
    /// only to mirror the source depth's descriptor.
    static func makeDepthLike(device: MTLDevice, _ tex: MTLTexture) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: tex.width, height: tex.height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.depth.previous")
        }
        t.label = "Illuminatorama.depth.previous"
        return t
    }

    /// `illumi_fs` specialised for one lane (DH-0140): `extended` selects the six-target output
    /// struct. The fragment carries a function constant now, so it can no longer be made by name alone.
    static func makeGBufferFragment(library: MTLLibrary, extended: Bool, interior: Bool = false) throws -> MTLFunction {
        let c = MTLFunctionConstantValues()
        var ext = extended
        c.setConstantValue(&ext, type: .bool, index: 11)
        var imap = interior
        c.setConstantValue(&imap, type: .bool, index: 12)
        return try library.makeFunction(name: "illumi_fs", constantValues: c)
    }

    /// The interior-mapping G-buffer pipeline for this lane (built once; nil if it fails to compile,
    /// in which case the kind draws with the ordinary pipeline — its flat emission).
    func interiorGBufferPipeline(extended: Bool) -> MTLRenderPipelineState? {
        if let p = extended ? gbs.gbufferPipelineInteriorExt : gbs.gbufferPipelineInterior { return p }
        guard let library = engine.library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = extended ? "Illuminatorama.gbuffer.interior.ext" : "Illuminatorama.gbuffer.interior"
        let vsC = MTLFunctionConstantValues()
        var pv = false
        vsC.setConstantValue(&pv, type: .bool, index: 10)
        do {
            d.vertexFunction = try library.makeFunction(name: "illumi_vs", constantValues: vsC)
            d.fragmentFunction = try Self.makeGBufferFragment(library: library, extended: extended, interior: true)
        } catch {
            Self.log.error("interior-mapping G-buffer: compile failed: \(error.localizedDescription)")
            return nil
        }
        d.colorAttachments[0].pixelFormat = .rgba16Float
        d.colorAttachments[1].pixelFormat = .rgba16Float
        d.colorAttachments[2].pixelFormat = .rgba16Float
        d.colorAttachments[3].pixelFormat = .rg16Float
        d.colorAttachments[4].pixelFormat = .r32Uint
        if extended { d.colorAttachments[5].pixelFormat = .rgba16Float }
        d.depthAttachmentPixelFormat = .depth32Float
        let p = try? device.makeRenderPipelineState(descriptor: d)
        if extended { gbs.gbufferPipelineInteriorExt = p } else { gbs.gbufferPipelineInterior = p }
        return p
    }

    /// DH-0140 — build one six-target G-buffer pipeline variant. Same attachments as the live PSO
    /// plus color(5) RGBA16F for `GBufferOut.material`.
    static func makeExtendedGBufferPipeline(device: MTLDevice, library: MTLLibrary,
                                                    prevVerts: Bool) throws -> MTLRenderPipelineState {
        let d = MTLRenderPipelineDescriptor()
        d.label = prevVerts ? "Illuminatorama.gbuffer.prevVerts.ext" : "Illuminatorama.gbuffer.ext"
        let vsC = MTLFunctionConstantValues()
        var pv = prevVerts
        vsC.setConstantValue(&pv, type: .bool, index: 10)
        d.vertexFunction = try library.makeFunction(name: "illumi_vs", constantValues: vsC)
        d.fragmentFunction = try makeGBufferFragment(library: library, extended: true)
        d.colorAttachments[0].pixelFormat = .rgba16Float
        d.colorAttachments[1].pixelFormat = .rgba16Float
        d.colorAttachments[2].pixelFormat = .rgba16Float
        d.colorAttachments[3].pixelFormat = .rg16Float
        d.colorAttachments[4].pixelFormat = .r32Uint
        d.colorAttachments[5].pixelFormat = .rgba16Float
        d.depthAttachmentPixelFormat = .depth32Float
        return try device.makeRenderPipelineState(descriptor: d)
    }

    /// DH-0140 — make the sixth target and the two extended pipelines exist for this frame, at the
    /// current internal size. Returns false (logged) if Metal refuses, in which case the frame runs
    /// the five-target pass and the lighting kernel stays on its live-lane constants.
    func ensureExtendedGBufferResources() -> Bool {
        if gbs.gbufferPipelineExt == nil || gbs.gbufferPipelinePrevVertsExt == nil {
            guard let library = engine.library else {
                Self.log.error("extended G-buffer: no shader library — staying on the 5-target pass")
                return false
            }
            do {
                gbs.gbufferPipelineExt = try Self.makeExtendedGBufferPipeline(device: device, library: library, prevVerts: false)
                gbs.gbufferPipelinePrevVertsExt = try Self.makeExtendedGBufferPipeline(device: device, library: library, prevVerts: true)
            } catch {
                Self.log.error("extended G-buffer: pipeline compile failed: \(error.localizedDescription) — staying on the 5-target pass")
                return false
            }
        }
        if let t = gbs.gbufferMaterial, t.width == width, t.height == height { return true }
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        d.pixelFormat = .rgba16Float
        d.width = width
        d.height = height
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            Self.log.error("extended G-buffer: material target allocation failed at \(self.width)×\(self.height)")
            return false
        }
        t.label = "Illuminatorama.gbuffer.material"
        gbs.gbufferMaterial = t
        return true
    }

    static func makeTargets(
        device: MTLDevice,
        internalW: Int, internalH: Int,
        outputW: Int, outputH: Int
    ) throws -> Targets {
        func make(label: String, format: MTLPixelFormat, w: Int, h: Int,
                  usage: MTLTextureUsage) throws -> MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = format
            d.width = w
            d.height = h
            d.usage = usage
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else {
                throw IlluminatoramaError.bufferAllocationFailed(label)
            }
            t.label = label
            return t
        }
        // All internal-pipeline targets are sized to internalW × internalH —
        // these are the textures every compute kernel except the tonemap
        // reads from and writes into. The tonemap's `outputTexture` is the
        // only thing at the smaller output size.
        let am = try make(label: "Illuminatorama.gbuffer.albedoMet",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let nr = try make(label: "Illuminatorama.gbuffer.normalRgh",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let em = try make(label: "Illuminatorama.gbuffer.emission",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let depth = try make(label: "Illuminatorama.depth",
                             format: .depth32Float, w: internalW, h: internalH,
                             usage: [.renderTarget, .shaderRead])
        // HDR composite and history need .sample because the tonemap kernel
        // bilinearly downsamples them into the output texture.
        let hdr = try make(label: "Illuminatorama.hdr",
                           format: .rgba16Float, w: internalW, h: internalH,
                           usage: [.shaderRead, .shaderWrite, .renderTarget])
        // Phase 4.11 — `.renderTarget` enables the additive forward
        // particles pass to draw directly into the composite. Compute
        // passes elsewhere still bind it via shaderRead/Write.
        let hdrComposite = try make(label: "Illuminatorama.hdrComposite",
                                    format: .rgba16Float, w: internalW, h: internalH,
                                    usage: [.renderTarget, .shaderRead, .shaderWrite])
        // RT diffuse (soft shadow + 1-bounce GI) buffer, full-res. The RT pass
        // writes here; `illumi_rt_denoise` bilateral-filters it into the composite.
        let rtDiffuse = try make(label: "Illuminatorama.rt.diffuse",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let velocity = try make(label: "Illuminatorama.velocity",
                                format: .rg16Float, w: internalW, h: internalH,
                                usage: [.renderTarget, .shaderRead])
        // Light-layer target — R32Uint so the full 32-bit bitfield round-trips
        // exactly (no half-float rounding). Written by the G-buffer fragment
        // shaders, read by the lighting kernel at [[texture(18)]].
        let layer = try make(label: "Illuminatorama.gbuffer.layer",
                             format: .r32Uint, w: internalW, h: internalH,
                             usage: [.renderTarget, .shaderRead])
        let historyA = try make(label: "Illuminatorama.historyA",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let historyB = try make(label: "Illuminatorama.historyB",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let taaResolved = try make(label: "Illuminatorama.taaResolved",
                                   format: .rgba16Float, w: internalW, h: internalH,
                                   usage: [.shaderRead, .shaderWrite])
        // AO and bloom run at half of the INTERNAL resolution (still oversampled
        // vs the output when SSAA is on).
        let halfW = max(1, internalW / 2)
        let halfH = max(1, internalH / 2)
        let ao = try make(label: "Illuminatorama.ao",
                          format: .r16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        // Phase 4.39 — SSAO denoiser intermediates.
        let aoFiltered = try make(label: "Illuminatorama.ao.filtered",
                                  format: .r16Float, w: halfW, h: halfH,
                                  usage: [.shaderRead, .shaderWrite])
        let aoHistA = try make(label: "Illuminatorama.ao.historyA",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        let aoHistB = try make(label: "Illuminatorama.ao.historyB",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        // Phase 4.39 — SSR denoiser intermediates.
        let ssrRaw = try make(label: "Illuminatorama.ssr.raw",
                              format: .rgba16Float, w: internalW, h: internalH,
                              usage: [.shaderRead, .shaderWrite])
        let ssrHistA = try make(label: "Illuminatorama.ssr.historyA",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let ssrHistB = try make(label: "Illuminatorama.ssr.historyB",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        // RT-GI temporal accumulation history (ping-pong, full-res).
        let rtGiHistA = try make(label: "Illuminatorama.rt.giHistoryA",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let rtGiHistB = try make(label: "Illuminatorama.rt.giHistoryB",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        // Phase 4.44 — SVGF denoiser: adaptive sample counts + à-trous buffers.
        let aoCount = try make(label: "Illuminatorama.svgf.aoCount",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        let ssrCount = try make(label: "Illuminatorama.svgf.ssrCount",
                                format: .r16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let giCount = try make(label: "Illuminatorama.svgf.giCount",
                               format: .r16Float, w: internalW, h: internalH,
                               usage: [.shaderRead, .shaderWrite])
        let giVar = try make(label: "Illuminatorama.svgf.giVariance",
                             format: .r16Float, w: internalW, h: internalH,
                             usage: [.shaderRead, .shaderWrite])
        let giVarB = try make(label: "Illuminatorama.svgf.giVarianceB",
                              format: .r16Float, w: internalW, h: internalH,
                              usage: [.shaderRead, .shaderWrite])
        let giAtrousA = try make(label: "Illuminatorama.svgf.giAtrousA",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let giAtrousB = try make(label: "Illuminatorama.svgf.giAtrousB",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        // Issue #65 — screen-space SSS buffers (full-res, idle unless sssStrength > 0).
        let sssDiffuse = try make(label: "Illuminatorama.sss.diffuse",
                                  format: .rgba16Float, w: internalW, h: internalH,
                                  usage: [.shaderRead, .shaderWrite])
        let sssBlurTmp = try make(label: "Illuminatorama.sss.blurTmp",
                                  format: .rgba16Float, w: internalW, h: internalH,
                                  usage: [.shaderRead, .shaderWrite])
        let sssBlurOut = try make(label: "Illuminatorama.sss.blurOut",
                                  format: .rgba16Float, w: internalW, h: internalH,
                                  usage: [.shaderRead, .shaderWrite])
        // Bloom mip pyramid (S1.2). Level 0 is half the internal resolution — the
        // same size and format the old single-gaussian target was, which is why
        // the tonemap's binding and sampler are untouched by this change.
        let bloomLevels = Self.bloomLevelCount(halfW: halfW, halfH: halfH)
        var bloomDown: [MTLTexture] = []
        var bloomUp: [MTLTexture] = []
        for level in 0..<bloomLevels {
            let (lw, lh) = Self.bloomLevelSize(halfW: halfW, halfH: halfH, level: level)
            bloomDown.append(try make(label: "Illuminatorama.bloom.down\(level)",
                                      format: .rgba16Float, w: lw, h: lh,
                                      usage: [.shaderRead, .shaderWrite]))
            // The SMALLEST level gets no up-chain buffer: the up pass is seeded by
            // reading `bloomDown.last` directly, so allocating one would be a
            // texture nothing ever writes.
            if level < bloomLevels - 1 {
                bloomUp.append(try make(label: "Illuminatorama.bloom.up\(level)",
                                        format: .rgba16Float, w: lw, h: lh,
                                        usage: [.shaderRead, .shaderWrite]))
            }
        }
        // Film halation at HALF res — the halo's orange core is only a few output pixels
        // wide. Allocated unconditionally so the tonemap always has a texture to bind at
        // fragment slot 8; the passes that fill it are skipped when halationIntensity == 0.
        // `bright` holds one scalar (the red-weighted exposure), `blurH` four (one per scale).
        let hb = try make(label: "Illuminatorama.halation.bright",
                          format: .r16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        let hbh = try make(label: "Illuminatorama.halation.blurH",
                           format: .rgba16Float, w: halfW, h: halfH,
                           usage: [.shaderRead, .shaderWrite])
        let hbv = try make(label: "Illuminatorama.halation.blurV",
                           format: .rgba16Float, w: halfW, h: halfH,
                           usage: [.shaderRead, .shaderWrite])
        // Final LDR output at OUTPUT resolution — the only texture sized to
        // what the SCNView shows. Tonemap kernel downsamples from internal
        // HDR + bloom into this each frame.
        let ldrDesc = MTLTextureDescriptor()
        ldrDesc.textureType = .type2D
        // sRGB so the tonemap can write linear + the GPU encodes on store; see
        // `makeLDRTexture`. sRGB targets disallow `.shaderWrite` (render-pass only).
        ldrDesc.pixelFormat = .bgra8Unorm_srgb
        ldrDesc.width = outputW
        ldrDesc.height = outputH
        ldrDesc.usage = [.shaderRead, .renderTarget]
        ldrDesc.storageMode = .private
        guard let ldr = device.makeTexture(descriptor: ldrDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("ldr")
        }
        ldr.label = "Illuminatorama.ldr"
        return Targets(
            albedoMet: am, normalRgh: nr, emission: em, depth: depth,
            hdr: hdr, ao: ao, hdrComposite: hdrComposite, rtDiffuse: rtDiffuse,
            velocity: velocity, layer: layer, historyA: historyA, historyB: historyB,
            taaResolved: taaResolved,
            bloomDown: bloomDown, bloomUp: bloomUp,
            halationBright: hb, halationBlurH: hbh, halationBlurV: hbv,
            ldr: ldr,
            aoFiltered: aoFiltered, aoHistoryA: aoHistA, aoHistoryB: aoHistB,
            ssrRaw: ssrRaw, ssrHistoryA: ssrHistA, ssrHistoryB: ssrHistB,
            rtGIHistoryA: rtGiHistA, rtGIHistoryB: rtGiHistB,
            aoSampleCount: aoCount, ssrSampleCount: ssrCount, giSampleCount: giCount,
            giVariance: giVar, giVarianceB: giVarB, giAtrousA: giAtrousA, giAtrousB: giAtrousB,
            sssDiffuse: sssDiffuse, sssBlurTmp: sssBlurTmp, sssBlurOut: sssBlurOut
        )
    }
}
