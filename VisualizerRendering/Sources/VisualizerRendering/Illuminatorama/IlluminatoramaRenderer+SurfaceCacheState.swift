import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - SurfaceCacheState

/// Surface cache (Lumen-style card atlas): cards, atlases, charts, GPU diff buffers.
/// Owned by the renderer as `scs`; every member keeps the name it had on the renderer.
@MainActor
final class SurfaceCacheState {
    var surfCachePipeline: MTLComputePipelineState?
    /// §3 endpoint — TLAS-traced cache-update (traces the per-frame-refit instance
    /// AS so a moved object's shadow/bounce-occlusion tracks). Used in place of
    /// `surfCachePipeline` when a TLAS is live (`rtTLASActive`); the soup path keeps
    /// the primitive-AS `surfCachePipeline`.
    var surfCacheTLASPipeline: MTLComputePipelineState?
    var surfCardBuffer: MTLBuffer?

    var surfTriCardBuffer: MTLBuffer?  // uint per triangle → card index
    var surfTriUVaBuffer: MTLBuffer?  // float4 (uvA.xy, uvB.xy)
    var surfTriUVcBuffer: MTLBuffer?  // float4 (uvC.xy, 0, 0)
    // P2 — per-card atlas RECT addressing (replaces the uniform tilesPerRow×tileSize
    // grid). `cardRect[card]` = (atlasX, atlasY, tileW, tileH) in atlas pixels;
    // `texelCard[y·atlasW + x]` = the card owning that texel (0xFFFFFFFF = gap).
    // Lets charts of different world extent get different-sized tiles, shelf-packed
    // into one atlas — the uniform grid is just the special case (all tiles equal).
    var surfCardRectBuffer: MTLBuffer?  // float4 per card (x,y,w,h px)
    var surfTexelCardBuffer: MTLBuffer?  // uint per atlas texel → card index

    var surfAtrousPipeline: MTLComputePipelineState?

    var surfCardRequestedBuffer: MTLBuffer?  // uint per card (1 = sampled this frame)

    var surfResidentCardCount: Int = 0   // ≤ surfCardCount; == it when budget disabled

    var surfSlotRect: [SIMD4<Float>] = []   // per slot: atlas rect (bake-time resident-card rects)
    var surfSlotCard: [Int] = []            // slot → occupant card index
    var surfCardSlot: [Int] = []            // card → slot (−1 = non-resident)
    let surfMaxSwapsPerFrame = 256          // bound the per-frame texelCard rewrite + anti-thrash
    // Phase 5 / A2 — eviction hysteresis. Per-card "heat": set to `surfStreamHeatFull`
    // when sampled, decremented each frame otherwise. A resident card is demotable only
    // once its heat hits 0 (unsampled for HeatFull consecutive frames), so a card
    // sampled intermittently (grazing rays, the moving box's trail) stays resident
    // instead of flipping in/out every frame — cuts the steady-state swap churn.
    var surfCardHeat: [Int] = []
    let surfStreamHeatFull =
        Int(ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_HEAT"] ?? "") ?? 4

    var surfCardCentroid: [SIMD3<Float>] = []   // bake-time world centroid per card
    var surfCardDirtyBuffer: MTLBuffer?  // uint per card (1 = re-light fresh)
    var surfTriCardCPU: [UInt32] = []       // triCard mirror (per soup triangle)
    var surfTriIsFrameB: [Bool] = []        // per triangle: B-half of its packed card?
    var surfSoupBaseCPU: [UInt32] = []       // per instance: first soup-triangle index
    var surfInstanceKind: [IlluminatoramaRenderer.MeshKind] = []    // per instance: mesh kind (for object soup)
    var surfObjectSoupByKind: [IlluminatoramaRenderer.MeshKind: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
    var surfIncrementalReady: Bool = false   // CPU maps valid for this topology

    var surfChartIncremental: IlluminatoramaRenderer.ChartIncremental?
    // Reused per-frame scratch for the chart incremental path's "which instances
    // moved" pass — sized once at bake time (when instance count is known) and
    // reset in place each frame, so the steady-state loop allocates nothing.
    var surfChartMovedScratch: [Bool] = []

    var surfReframeTriPipeline: MTLComputePipelineState?
    var surfReframeChartPipeline: MTLComputePipelineState?
    var surfReframeDeformPipeline: MTLComputePipelineState?
    var surfGPUDiffTriOwnerBuffer: MTLBuffer?  // uint per soup tri → instance
    var surfGPUDiffTriFlagsBuffer: MTLBuffer?  // uint per soup tri: SC_FLAG_* bits
    var surfGPUDiffObjVertsBuffer: MTLBuffer?  // float4 ×3 per soup tri (object space)
    var surfGPUDiffTriCount: Int = 0
    var surfGPUDiffChartOwnerBuffer: MTLBuffer?  // int32 per card (−1 = static)
    var surfGPUDiffChartFrameBuffer: MTLBuffer?  // float4 ×4 per card (object frame)
    // ── Phase D — deforming meshes, GPU-resident ──
    /// Vertex-index triple per GLOBAL soup triangle (topology constant per bake);
    /// the deform kernel reads these + the LIVE packed_float3 position buffer.
    var surfGPUDiffDeformIdxBuffer: MTLBuffer?

    var surfGPUDiffDeformDispatches: [IlluminatoramaRenderer.SurfDeformDispatch] = []
    /// 4×uint written by the kernels via relaxed atomics ([0] moved instances,
    /// [1] dirtied cards, [2] deformed instances, [3] spare), read + zeroed by the CPU
    /// for the `recordSurfCacheStats` sidecar. RING-buffered (DH-0537 #1): with
    /// `maxFramesInFlight == 2` the previous frame's kernel can still be incrementing while
    /// this frame's CPU reads and zeroes, so one buffer was a read-during-write race. A slot
    /// is reused every `maxFramesInFlight` frames, and the in-flight semaphore guarantees the
    /// frame that last wrote it has COMPLETED before this frame's encode begins.
    var surfGPUDiffStatsRing: [MTLBuffer] = []

    // Phase C — a deforming (DynamicMesh / GPU-fed) mesh contributed surface cards
    // this topology. Its cards are re-framed from LIVE vertices every frame
    // (`objectTriangleSoup()` is empty for its private vertex buffer, so it's
    // otherwise excluded from the cache), and the update kernel must honour the
    // per-card dirty flags (α=1) just as for rigid incremental invalidation.
    var surfHasDeformingCards: Bool = false
    /// Dev A/B override (`VIZ_ILLUMI_NO_DEFORM_CARDS=1`) — freeze deforming cards
    /// at their build pose (pre-Phase-C behaviour) so the catch/cast win is
    /// directly comparable. Off by default; the re-frame is on whenever a
    /// deforming mesh drives the cache.
    let noDeformCardsOverride =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_DEFORM_CARDS"] == "1"
    var surfAtlasA: MTLTexture?
    var surfAtlasB: MTLTexture?
    /// DH-0622 — the INDIRECT term alone, ping-ponged in lockstep with the full atlas. The
    /// full atlas (direct + indirect) is only the update kernel's multi-bounce feedback; every
    /// consumer (GI, reflection, glass) shades its hit itself and reads only this.
    var surfIndirectAtlasA: MTLTexture?
    var surfIndirectAtlasB: MTLTexture?
    // Phase 5 / B1 — cache-domain denoiser output. NOT ping-ponged: recomputed
    // each frame from the just-updated current INDIRECT atlas (display-only; the EMA
    // feedback keeps reading the raw atlases). Bound to the GI/reflection consumers
    // in place of the current indirect atlas when `surfaceCacheDenoise` is on.
    var surfAtlasDenoised: MTLTexture?
    var surfUseAtlasA: Bool = true
    var surfCacheUniformBuffer: MTLBuffer!
    var surfCardCount: Int = 0
    /// Default per-card atlas tile size — sized for the room's big planar cards.
    var surfTileSize: Int = 64
    /// Tile size of the CURRENTLY-registered card set. Per-triangle micro-cards
    /// (Phase 4.38) pass a much smaller tile (a triangle doesn't need 64²), so
    /// the atlas stays bounded; the room keeps 64. Set by `setSurfaceCacheCards`.
    var surfActiveTileSize: Int = 64
    var surfTilesPerRow: Int = 1
    var surfAtlasW: Int = 1
    var surfAtlasH: Int = 1
    var surfFrameSeed: UInt32 = 0
    /// The (TLAS topology, triangle budget) the cards were last baked — or skipped — for. Lets a
    /// host turn the cache on over an already-built TLAS and get cards on the next frame, without
    /// an over-budget scene re-attempting the bake every frame (DH-0622).
    var surfCacheBakeKey: Int?
    /// Warm-up updates still owed by the latest bake or enable (DH-0622) — see `surfaceCacheWarmIterations`.
    var surfCacheWarmPending = 0

    var surfCacheTLASCurvePipeline: MTLComputePipelineState?

    var surfCacheSoupCurvePipeline: MTLComputePipelineState?

    // ── DH-0653: surface-cache hit/miss + atlas reuse counters ────────────────
    //
    // Two atomic counter buffers the kernels add to only while the stats sidecar is
    // set: `surfHitStatsBuffer` (8 slots, `illumi_rt_lighting_tlas` — what each GI /
    // reflection triangle hit was served by) and `surfAtlasStatsBuffer` (2 slots, the
    // update kernel — texels EMA-reused vs refreshed at α = 1). Each is drained at the
    // NEXT frame's encode, after the in-flight frame completes, and written as one
    // `hits:` / `atlas:` line. The DH-0622 trade ("cheaper than re-shade" → "richer
    // than re-shade") is the `cached` share of the `hits:` line.
    var surfHitStatsBuffer: MTLBuffer?
    var surfAtlasStatsBuffer: MTLBuffer?
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// Surface-cache update pipelines (RT-gated), uniforms, GPU-diff re-frame + denoise kernels and the stats ring.
    @MainActor static func buildSurfaceCachePipelines(into s: SurfaceCacheState, cache: SimPipelineCache, device: MTLDevice, library: MTLLibrary) throws {
        let curvesOff = Self.curvesOffConstants()
        // ── Surface-cache update pipeline (RT-gated) ─────────────────
        // Both kernels declare the kSCCurvesEnabled function constant (#60 item 7),
        // so both MUST be specialized via constantValues even for the base (false)
        // variant — Metal refuses a plain makeFunction(name:) pipeline for any
        // constant-declaring function. `curvesOff` was built above.
        var builtSurfCachePipeline: MTLComputePipelineState? = nil
        var builtSurfCacheTLASPipeline: MTLComputePipelineState? = nil
        if device.supportsRaytracing,
           let scFn = try? library.makeFunction(name: "illumi_surfcache_update",
                                                constantValues: curvesOff) {
            builtSurfCachePipeline = try? device.makeComputePipelineState(function: scFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        if device.supportsRaytracing,
           let scTLASFn = try? library.makeFunction(name: "illumi_surfcache_update_tlas",
                                                    constantValues: curvesOff) {
            builtSurfCacheTLASPipeline = try? device.makeComputePipelineState(function: scTLASFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        s.surfCachePipeline = builtSurfCachePipeline
        s.surfCacheTLASPipeline = builtSurfCacheTLASPipeline
        guard let scUB = device.makeBuffer(length: MemoryLayout<SurfCacheUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("surfCacheUniforms")
        }
        scUB.label = "Illuminatorama.surfcache.uniforms"
        s.surfCacheUniformBuffer = scUB
        // #60 item 1 + Phase D — GPU-side incremental-invalidation kernels (plain
        // compute, no RT dependency; nil leaves the CPU re-frame path in charge).
        s.surfReframeTriPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_tri", device: device)
        s.surfReframeChartPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_chart", device: device)
        s.surfReframeDeformPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_deform", device: device)
        // Phase 5 / B1 — cache-domain denoiser (plain compute, no RT dependency;
        // nil leaves consumers reading the raw atlas).
        s.surfAtrousPipeline =
            cache.pipelineState(name: "illumi_surfcache_atrous", device: device)
        var statsRing: [MTLBuffer] = []
        for slot in 0..<IlluminatoramaRenderer.maxFramesInFlight {
            guard let b = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 4,
                                            options: .storageModeShared) else { continue }
            b.label = "Illuminatorama.surfcache.gpuDiffStats.\(slot)"
            memset(b.contents(), 0, b.length)
            statsRing.append(b)
        }
        s.surfGPUDiffStatsRing = statsRing
    }
}
