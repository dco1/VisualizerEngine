import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - GIKnobs

/// Stored state for the renderer's `gi` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `GIKnobs` (`kGI`) and forwards each name to it from
/// `IlluminatoramaRenderer+GIKnobs.swift`; callers see no difference.
@MainActor
final class GIKnobs {
    typealias DebugTerm = IlluminatoramaRenderer.DebugTerm
    typealias MeshKind = IlluminatoramaRenderer.MeshKind

    // ── Daydream DH-1304 — the per-term AOV capture ──────────────────────────────────────────
    /// Capture every lighting term of every frame into its own running mean (`IlluminatoramaAOV`,
    /// `Shaders/IlluminatoramaAOV.h` for the set and the contract). DEBUG / MEASURE ONLY: while
    /// false — every host's default — no variant, no texture and no binding exist, and the lighting
    /// kernels are byte-for-byte the pre-DH-1304 code (every tag is behind a function constant).
    /// While true the frame runs the `-aov` variants of `illumi_lighting` and
    /// `illumi_rt_lighting_tlas` (compiled blocking on first use) and accumulates
    /// `aovLightingTexture` as a per-pixel mean over the frames since the last reset. The mean
    /// restarts whenever the GI running mean does (`resetRTGIAccumulation`, the still's held-out
    /// warm-up / cache-build frames) and on `resetAOVAccumulation()`. Σ(lighting slices) == beauty
    /// per pixel on every frame — `HouseRenderBridgeGPUTests_AOV` is the gate.
    /// Only the TLAS lane is tagged: the soup kernel (`illumi_rt_lighting`) and the isolation
    /// `debugTerm` views write no AOVs.
    var aovCaptureEnabled: Bool = false
    /// Frames in the current mean (0 ⇒ nothing captured since the last reset).
    var aovAccumulatedFrames: Int = 0
    /// Multi-bounce AO (Jimenez 2016 fit) on the lighting kernel's AO term. OFF by default => byte-identical.
    /// Rides `filmMedium.z`. DH-0953.
    var aoMultiBounce: Bool = false
    /// Test-only: offsets the progressive sequence's streams so two otherwise identical settles
    /// sample independently (production sampling is deterministic). 0 = production.
    var aoAccumulationScramble: UInt32 = 0
    /// Daydream DH-1410 — with `aoAccumulationEnabled` AND `taaStillAccumulation` (the averager), do not keep
    /// the AO's own running mean: each frame's occlusion is filtered (at the converged ½ strength) and handed
    /// to the lighting, and the colour resolve's frozen-camera average is the one mean — the AO twin of
    /// `rtGIStillPerFrame`. Default `false`: byte-identical for every host.
    var aoStillPerFrame: Bool = false
    /// Whether THIS frame's AO came from the accumulator (the selector's switch).
    var aoAccumulatorOwnsAO: Bool = false
    /// Indirect rays per atlas texel in the update pass (multi-bounce gather).
    var surfCacheIndirectRays: Int = 4
    /// Temporal EMA blend for the cache update — fraction of the new radiance
    /// folded in each frame. Lower = smoother/slower convergence.
    var surfCacheAlpha: Float = 0.1
    /// Extra α = 1 cache updates run on the frame after cards bake or the cache turns on (DH-0622),
    /// so a still that switches the cache on starts from a filled atlas instead of an empty one —
    /// each update carries the light one bounce further. 0 (default) = none.
    var surfaceCacheWarmIterations: Int = 0
    /// DH-0849 — an indirect ray in the cache update that strikes the BACK of a card (its triangle's
    /// winding normal faces along the ray) brings back no light instead of that card's front-side
    /// irradiance. Without it, wall bands hidden inside a closed plenum (above a ceiling deck, below a
    /// floor) read the deck's room-side light off its back and glow, and GI hits at the crease pick
    /// that up as a bright line. Changing it takes effect on the next update (the EMA re-converges).
    var surfaceCacheBackFaceGuard: Bool = true
    /// **Defocus-aware shading rate for the lighting pass** (opt-in, VZ-0197; 0 = off = byte-identical
    /// — the default lighting variant never reads the gate and no extra pass is encoded).
    ///
    /// With depth of field on, a 16×16 tile whose geometry is ALL at least this far out of focus
    /// (CoC radius, internal px, under the DOF's own lens and clamps; floored at 4) has each
    /// one-surface 2×2 quad (four geometry pixels whose view depths agree within 2 %) lit ONCE, at
    /// its top-left pixel, and that radiance written to all four; every other pixel is lit exactly as
    /// before. The lens then spreads each of those pixels over a disc ≥ π·gate² px, so the 2×2
    /// replication (a 2-px box, σ 0.5 px, and a ½-px shift) widens the blur against the disc's σ of
    /// ≥ gate/2 px by < 1 % in σ at an 8 px gate, 1.4 % at 6. It point-samples the quad, so detail
    /// within ~0.05 cycles/px of Nyquist can alias below the disc's cut-off — texture-filtered
    /// G-buffers carry little there; the look diff is the check. Two dispatches of the same kernel (a coarse variant, a thread
    /// per quad, and a skip variant) whose pixel sets partition the frame, plus a depth-only tile
    /// reduction (`illumi_dof_shading_tiles`). Off whenever `dofEnabled` is off. Lighting-pass
    /// consumers that must see every pixel lit individually (SSR, RT reflections, SSS) see the
    /// replicated values in those tiles — acceptable only because the DOF blurs them past a disc.
    var lightingDefocusShadingMinCoC: Float = 0
    /// DH-0889 — jitter each window portal's shadow-map projection by a sub-texel Halton offset
    /// per TAA frame, so an accumulating still supersamples the occluder's edge instead of
    /// printing one sawtooth per texel. Default OFF: it re-renders the portal slices every frame,
    /// which a live canvas's static-slice reuse would otherwise skip. Only the shadow-MAP path
    /// uses it; traced portal visibility (`rtAreaShadowRays > 0`) has no texels to jitter.
    var areaShadowJitterEnabled: Bool = false
    var debugTerm: DebugTerm = .normal
    // ── Phase 2 post-FX knobs ────────────────────────────────────────
    /// Screen-space ambient occlusion intensity (0 = disabled).
    var ssaoIntensity: Float = 0.85
    /// World-space hemisphere radius for SSAO samples, in metres.
    var ssaoRadius: Float = 0.4
    /// DH-0441 — outer radius (metres) of the wide AO ring, a second GTAO march whose
    /// occluders are weighted to the annulus beyond `ssaoRadius`. See IlluminatoramaSSAO.metal.
    var ssaoFarRadius: Float = 0.5
    /// DH-0441 — strength of the wide AO ring. 0 (default) = OFF, the ring is never marched
    /// and the AO field is byte-identical.
    var ssaoFarIntensity: Float = 0
    /// Screen-space reflection intensity (0 = disabled).
    var ssrIntensity: Float = 0.7
    /// Max view-space ray length for SSR marches, in metres.
    var ssrMaxDistance: Float = 18.0
    /// View-space "thickness" tolerance for the depth-crossing hit test.
    var ssrThickness: Float = 0.6
    /// Cap on march steps per SSR ray.
    var ssrMaxSteps: UInt32 = 48
    /// **The photo-lane G-buffer (DH-0140).** When true the G-buffer pass runs the six-target
    /// pipeline variant and writes `GBufferOut.material` — a per-pixel material payload the
    /// five-target live pass has no room for (its one spare scalar is a discrete class tag), and
    /// the lighting kernel reads it. The sixth target and its two pipeline variants are
    /// allocated / compiled on FIRST use and dropped on resize, so a host that never sets this —
    /// every Visualizer scene, Daydream's live canvas — is byte-identical and pays nothing.
    /// Payloads so far: per-material clearcoat roughness (slice 1). Reserved lanes:
    /// transmission/alpha, per-texel grain tangent (DH-0478). A host's photo / export lane sets
    /// this for the still and clears it on the way out.
    var extendedGBufferEnabled: Bool = false
    /// True while THIS frame's G-buffer pass wrote the sixth target — the flag AND the target
    /// allocation both succeeded. Diagnostics / gates.
    var extendedGBufferActive: Bool = false
    /// **Edge-aware SPATIAL filter on the half-res AO** (depth/normal-weighted
    /// blur). Intra-frame, so it cannot ghost — this is the half that makes raw
    /// 16-sample SSAO stop reading as grain. Default ON.
    ///
    /// Note the pass is still ENCODED when `ssaoTemporalEnabled` is on, because
    /// the temporal kernel reads `aoFilteredTexture` as its input; with this flag
    /// off the kernel passes the raw AO through instead of filtering it. See
    /// `ssaoSpatialRuns`.
    var ssaoSpatialEnabled: Bool = true
    /// **Velocity-reprojected TEMPORAL accumulation of the AO** (up to 32 frames).
    /// Default OFF: the reprojection uses the velocity of the pixel it is AT, but
    /// AO is a function of that pixel's NEIGHBOURHOOD — a mover's contact darkening
    /// lands on a static receiver whose velocity is zero, so the history reprojects
    /// onto itself and trails behind the mover. Neither guard catches it (the
    /// disocclusion term reads the same zero velocity; the 3×3 ±1.5σ clamp has a
    /// σ floor of 0.03). Fine for a static or slow camera; opt in per scene.
    var ssaoTemporalEnabled: Bool = false
    /// History retention weight for SSAO temporal (0 = all current, 1 = frozen).
    /// 0.90 gives ~10-frame effective sample count with fast transient recovery.
    var ssaoTemporalBlend: Float = 0.90
    /// **Velocity-reprojected TEMPORAL accumulation of the SSR gather.** Kills
    /// shimmer on rough/medium surfaces; smears a reflection of anything that moves
    /// across a static reflector, for the same structural reason as the AO twin.
    /// Default OFF. Off = raw single-sample gather goes straight to composite.
    ///
    /// SSR has NO spatial stage (gather → temporal → composite), so there is no
    /// `ssrSpatialEnabled` to pair with this.
    var ssrTemporalEnabled: Bool = false
    /// History retention weight for SSR temporal. 0.85 balances lag vs noise.
    var ssrTemporalBlend: Float = 0.85
    // ── DH-0534 — VISIBILITY CULLING, the renderer-side wiring ──────────────────
    // The logic and its correctness argument live in IlluminatoramaVisibilityCull.swift: a group
    // whose world bounds lie wholly outside one clip half-space of the matrix a pass rasterises
    // with cannot produce a fragment there, so skipping its draw changes no pixel. This is the
    // part DH-0871 lost to an accidental `git checkout` and nothing had called since — the logic
    // was committed (0718f54) with zero call sites, so every pass still drew every group.
    /// Master switch. A pure skip, so ON by default; the lever is for A/B and for a host that
    /// wants the old draw-everything behaviour back.
    var visibilityCullingEnabled = true
    /// Mesh kinds the CAMERA does not see: the raster passes (the G-buffer and the cascade / spot / point shadow maps) skip their
    /// groups, but the ray-tracing structure is built from the same groups and keeps them — so an instance of a hidden kind
    /// still lights the scene (path-traced GI, emission) and still appears in RAY-TRACED reflections and refractions.
    /// This is how a studio's light cards work (`IlluminatoramaStudio`): a polished piece mirrors a softbox that is nowhere in
    /// frame. A kind needs a mesh of its own (`setMesh`), since a group is the unit that is skipped.
    var cameraHiddenKinds: Set<MeshKind> = []
    /// Draw calls each raster pass issued and skipped, cumulative since init. Take deltas.
    var visibilityCullStats = IlluminatoramaVisibilityCullStats()
    // ── Incremental invalidation (default-on, per-triangle TLAS path) ──
    // When `surfaceCacheIncremental` is on and the topology is stable, each frame
    // the moved instances' cards are re-framed in place + flagged dirty so the
    // update kernel re-lights only them (α=1) while stationary cards keep their
    // accumulated multi-bounce. See docs/illuminatorama/surface-cache-incremental-
    // invalidation.md. DEFAULT ON: it self-gates to the per-triangle TLAS path
    // (`surfIncrementalReady`) and is a no-op on chart/soup/static scenes (a fully
    // static frame diffs to zero moved instances → zero re-framing), so enabling it
    // by default only helps animated TLAS scenes. `VIZ_SURFCACHE_INCREMENTAL=0` is
    // the kill-switch. The default lane re-frames on the GPU, on this frame's command
    // buffer (hazard-tracked); the CPU fallback rewrites `surfCardBuffer` in place and
    // first waits for the in-flight frame (`awaitInFlightFrameBeforeCPUCacheWrite`,
    // DH-0537 #1) — the pool is 2, so an unguarded in-place write would race.
    var surfaceCacheIncremental: Bool = true
    // Phase 5 / B1 — cache-domain à-trous denoiser. Spatially filters the radiance
    // atlas (same-card-guarded, variance-guided) before the GI/reflection consumers
    // read it, so a freshly-reset card resolves its Monte-Carlo noise in ~1 frame
    // instead of ~20. Display-only (the EMA feedback reads the raw atlas), so it
    // accelerates convergence without changing the fixed point. DEFAULT OFF while
    // it's staged in; `VIZ_ILLUMI_SURFCACHE_DENOISE=1` turns it on.
    // See docs/illuminatorama/phase5-radiance-cache-streaming.md.
    var surfaceCacheDenoise: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_DENOISE"] == "1"
    // Phase 5 / A (streaming) — residency feedback. When on, the GI/reflection
    // cache hits mark `cardRequested[card]=1`; the host drains the buffer the NEXT
    // frame (after waiting for the in-flight one) to log the per-frame WORKING SET size — the data
    // that sizes the atlas budget and that A1's residency pass will key off. Plain
    // instrumentation for now (no behavior change). `VIZ_ILLUMI_SURFCACHE_FEEDBACK=1`.
    var surfaceCacheFeedback: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_FEEDBACK"] == "1"
    // Phase 5 / A0 (streaming) — card budget. >0 caps the number of RESIDENT cards
    // (atlas sized to the budget, not the scene); cards beyond it get no atlas slot,
    // and a GI/reflection hit on a non-resident card falls back to emission + ambient
    // (never black). Static bake-order residency for A0 (feedback-driven priority is
    // A1). 0/unset ⇒ disabled (every card resident — today's behaviour).
    // `VIZ_ILLUMI_SURFCACHE_CARD_BUDGET`.
    var surfCardBudget: Int =
        Int(ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_CARD_BUDGET"] ?? "") ?? 0
    // Phase 5 / A1 (streaming) — DYNAMIC feedback-driven residency. When on (with a
    // budget), a per-frame host pass reassigns which cards occupy the fixed atlas
    // SLOTS (= A0's bake-time resident-card rects) based on the `cardRequested`
    // working set: a sampled-but-non-resident card is promoted into a slot freed by
    // an unsampled resident card. Mutates `texelCard` + `cardRect` + `dirty` in place
    // (after `awaitInFlightFrameBeforeCPUCacheWrite`, same class as the incremental writes) — NO
    // kernel addressing change (the kernels still read texel→card / card→rect, and
    // A0's zero-rect-⇒-fallback path already handles a non-resident card).
    // `VIZ_ILLUMI_SURFCACHE_STREAM=1` (implies feedback). Default off.
    var surfaceCacheStreaming: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_STREAM"] == "1"
    // Phase 5 / A4 — importance-weighted residency (the tractable core of per-card
    // LOD). Under eviction pressure (budget < working set), keep the NEAR cards (large
    // screen footprint) over far ones: promote the most-important requested cards
    // first, demote the least-important cold slots first. Importance = camera proximity
    // from the card's bake-time centroid. (Full variable-slot-size LOD — smaller atlas
    // tiles for far cards — is the deferred bin-packing follow-on; this prioritises
    // WHICH cards hold the fixed slots, not slot size.) `VIZ_ILLUMI_SURFCACHE_LOD=1`.
    var surfaceCacheLOD: Bool =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_LOD"] == "1"
    /// Instances the last card bake left cardless (DH-1314) — read-only diagnostic.
    var surfaceCacheCardlessInstances: Int = 0
    /// Surface-cache (P1c) caps for the TLAS path. The cache allocates one
    /// per-triangle micro-card over the INSTANCE-EXPANDED soup (count =
    /// Σ group.count · meshTris), which can dwarf the per-mesh `estTriangles`
    /// used to gate live RT — so it gets its own cap. Past it the TLAS still
    /// runs (RT lighting intact); only the cache read is skipped.
    ///
    /// A host that turns the cache on for a real scene raises it: a furnished house is well past
    /// 60 000 (4000 Sunset ~154 k), where the cache is skipped and the frame is identical to cache
    /// off. Read at each TLAS topology build, so set it before the build that should bake cards.
    /// `VIZ_ILLUMI_SURFCACHE_MAXTRIS` overrides the default — an instrument, like
    /// `VIZ_ILLUMI_RT_GROUPCAP` (DH-0622 measured 0 cached hits on the ~90 k Debug Tester without it).
    var surfaceCacheMaxTriangles: Int =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_MAXTRIS"].flatMap { Int($0) } ?? 60_000
    /// Atlas texels per side of a per-triangle card tile, read at each card bake. A large triangle's
    /// texels span tens of cm at 6, which is what DH-0849 (light along junctions) is being A/B'd against.
    var surfaceCachePerTriangleTileSize: Int = 6
    /// DH-1314 — how the cache update treats a bounce ray that lands on a CARDLESS mesh
    /// (`IlluminatoramaMesh.surfaceCacheCards = false`): true (default) re-shades it once — its
    /// own sun plus the gathering texel's indirect, times its albedo; false brings back nothing
    /// (a black occluder — the A/B arm). Inert in a scene with no cardless mesh.
    var surfaceCacheCardlessReshade: Bool = true
}
