import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

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
}
