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

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// G-buffer, superquadric-impostor pipelines and the opaque depth state (init order step 1).
    @MainActor static func buildGBufferPipelines(into s: GBufferState, device: MTLDevice, library: MTLLibrary) throws {
        // G-buffer render pipeline (vertex + fragment).
        let pdesc = MTLRenderPipelineDescriptor()
        pdesc.label = "Illuminatorama.gbuffer"
        // Issue #65 — `illumi_vs` now references function_constant(10) kUsePrevVerts,
        // so even the BASE variant must be specialized (a plain makeFunction(name:)
        // aborts at pipeline validation for any referenced-but-unset constant). The
        // base sets it false → byte-identical to the old behaviour; the prev-verts
        // pipeline below sets it true and gains the buffer(5) per-vertex prev read.
        let vsBaseConsts = MTLFunctionConstantValues()
        var userPrevVertsFalse = false
        vsBaseConsts.setConstantValue(&userPrevVertsFalse, type: .bool, index: 10)
        pdesc.vertexFunction = try library.makeFunction(name: "illumi_vs",
                                                        constantValues: vsBaseConsts)
        pdesc.fragmentFunction = try Self.makeGBufferFragment(library: library, extended: false)
        pdesc.colorAttachments[0].pixelFormat = .rgba16Float
        pdesc.colorAttachments[1].pixelFormat = .rgba16Float
        pdesc.colorAttachments[2].pixelFormat = .rgba16Float
        // Phase 2.7 — velocity attachment (screen-space motion in UV space).
        pdesc.colorAttachments[3].pixelFormat = .rg16Float
        // Light-layer attachment (R32Uint) — carries the per-fragment layer bitfield.
        pdesc.colorAttachments[4].pixelFormat = .r32Uint
        pdesc.depthAttachmentPixelFormat = .depth32Float
        // No vertex descriptor — the shader reads from a device-buffer of
        // Vertex via [[vertex_id]]. Simpler than wiring a vertex descriptor
        // and matches the Swift/Metal struct mirror more directly.
        s.gbufferPipeline = try device.makeRenderPipelineState(descriptor: pdesc)

        // Issue #65 — second G-buffer pipeline with kUsePrevVerts = true. Reuses
        // the SAME descriptor (identical attachment formats / fragment); only the
        // specialized vertex function differs. Bound for deforming GPU-mesh draws.
        let vsPrevConsts = MTLFunctionConstantValues()
        var usePrevVertsTrue = true
        vsPrevConsts.setConstantValue(&usePrevVertsTrue, type: .bool, index: 10)
        pdesc.label = "Illuminatorama.gbuffer.prevVerts"
        pdesc.vertexFunction = try library.makeFunction(name: "illumi_vs",
                                                        constantValues: vsPrevConsts)
        s.gbufferPipelinePrevVerts = try device.makeRenderPipelineState(descriptor: pdesc)

        // Perfect analytic superquadric impostor — same G-buffer attachment
        // formats so it renders into the SAME G-buffer pass/encoder (no extra
        // pass, no extra clear). The fragment writes analytic depth via
        // `[[depth(less)]]`. Optional: if the functions are missing the impostor
        // simply doesn't draw (the proxy still covers RT), so don't fail init.
        if let sqVS = library.makeFunction(name: "illumi_superquadric_impostor_vs"),
           let sqFS = library.makeFunction(name: "illumi_superquadric_impostor_fs") {
            let sqdesc = MTLRenderPipelineDescriptor()
            sqdesc.label = "Illuminatorama.superquadricImpostor"
            sqdesc.vertexFunction = sqVS
            sqdesc.fragmentFunction = sqFS
            sqdesc.colorAttachments[0].pixelFormat = .rgba16Float
            sqdesc.colorAttachments[1].pixelFormat = .rgba16Float
            sqdesc.colorAttachments[2].pixelFormat = .rgba16Float
            sqdesc.colorAttachments[3].pixelFormat = .rg16Float
            sqdesc.colorAttachments[4].pixelFormat = .r32Uint
            sqdesc.depthAttachmentPixelFormat = .depth32Float
            s.superquadricImpostorPipeline =
                try? device.makeRenderPipelineState(descriptor: sqdesc)
        }


        let ddesc = MTLDepthStencilDescriptor()
        ddesc.depthCompareFunction = .less
        ddesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: ddesc) else {
            throw IlluminatoramaError.pipelineCreationFailed("depth-stencil state")
        }
        s.depthState = ds
    }

    /// Warm the init compute pipelines concurrently (WWDC23 #10127 async compilation) so a cold pipeline cache compiles in parallel.
    @MainActor static func precompileInitPipelines(cache: SimPipelineCache, device: MTLDevice) {
        // illumi_lighting is function-constant–specialized (WWDC23 #10127). Build the
        // variant matching the renderer's current feature flags; other variants are
        // compiled lazily by `currentLightingPipeline()` when a flag toggles, and this
        // init-time variant doubles as the fallback if a later build ever fails.
        // #60 task 4 — warm the init compute pipelines concurrently (WWDC23
        // #10127 async compilation applied to startup) so a cold pipeline cache
        // compiles in parallel instead of 26 serial blocking calls. The guards
        // below then hit the warm cache. (`illumi_lighting` uses the specialized
        // variant path, so it's compiled by the guard, not here.)
        cache.precompile([
            "illumi_bloom_down", "illumi_bloom_prefilter", "illumi_bloom_up",
            "illumi_dfg_bake", "illumi_exposure_estimate",
            "illumi_halation_blur_h", "illumi_halation_blur_v", "illumi_halation_threshold",
            "illumi_irradiance_bake",
            "illumi_mesh_build_adjacency", "illumi_mesh_synth", "illumi_particles_step",
            "illumi_prefilter_bake", "illumi_repack_pos_norm", "illumi_rt_denoise",
            "illumi_rt_gi_temporal", "illumi_rtao_tlas", "illumi_ssao", "illumi_ssao_spatial",
            "illumi_ssao_temporal", "illumi_ssr_composite", "illumi_ssr_gather",
            "illumi_ssr_temporal", "illumi_surfcache_atrous",
            "illumi_surfcache_reframe_chart",
            "illumi_surfcache_reframe_deform", "illumi_surfcache_reframe_tri",
            "illumi_svgf_atrous", "illumi_svgf_variance", "illumi_taa_resolve",
        ], device: device)
    }

    /// The init-time `illumi_lighting` variant; doubles as the fallback if a later variant build fails.
    @MainActor static func buildLightingPipeline(into s: GBufferState, cache: SimPipelineCache, device: MTLDevice,
                                              ibl: Bool, shadow: Bool, dfg: Bool) throws {
        let (initLightingKey, initLightingConstants) = Self.lightingFeatureConstants(
            ibl: ibl, shadow: shadow, dfg: dfg)
        guard let lighting = cache.pipelineState(name: "illumi_lighting", device: device,
                                                 constants: initLightingConstants,
                                                 variantKey: initLightingKey) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_lighting")
        }
        s.lightingPipeline = lighting
    }
}
