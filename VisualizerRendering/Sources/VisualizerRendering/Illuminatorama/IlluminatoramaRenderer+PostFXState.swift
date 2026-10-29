import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - PostFXState

/// Post chain: TAA history, bloom, halation, glare, tonemap, exposure, colour LUT, present ring.
/// Owned by the renderer as `pfx`; every member keeps the name it had on the renderer.
@MainActor
final class PostFXState {
    var glareFitKey: (texelRad: Float, levels: Int, model: IlluminatoramaGlareSpread.Model)?
    /// DH-0998 — the far-field glare tail: tan of the half field of view (x, y) this frame, its
    /// output (rgb = the tail's glare, a = the in-frame share of each pixel's light it scatters),
    /// and its pipeline (built on first use: only an export on the physical lane ever runs it).
    var glareTailTanHalf: SIMD2<Float> = SIMD2(1, 1)
    var glareTailTexture: MTLTexture?

    var easedExposure: Float = 1.0
    var easedBloomThreshold: Float = 1.0
    var easedBloomIntensity: Float = 0.6
    var easedChromaticAberration: Float = 0
    var easedFringe: Float = 0
    var easedFringeTint: SIMD3<Float> = SIMD3(0.62, 0.12, 0.92)
    var easedSphericalAberration: Float = 0
    var easedHalationIntensity: Float = 0
    var easedHalationThreshold: Float = 0.6
    var easedHalationRadius: Float = 0.025
    var easedHalationTint: SIMD3<Float> = SIMD3(1.0, 0.20, 0.07)
    var lastPostFXEaseTime: CFTimeInterval = 0

    /// One-shot latch for the coarse-animation-clock warning in `render()` — a host that feeds a
    /// boot-relative clock would otherwise emit one line per frame forever.
    var warnedCoarseAnimationClock = false
    /// One-shot latch for the "wind armed but nothing opted in" warning in `uploadInstances`.
    var warnedWindWithoutOptIn = false

    // ── Presented-buffer pool (overlay↔SceneKit race fix) ──────────────────────
    //
    // `outputTexture` is sampled by SceneKit on its OWN command queue (the
    // overlay binds it as `displayScene.background.contents`), with no
    // cross-queue sync to this renderer's tonemap write. Writing the presented
    // texture in place therefore races SceneKit's read: a heavy scene whose
    // tonemap is still in flight when SceneKit composites is sampled mid-write
    // and reads as zeros (a transparent / white frame). A light scene wins the
    // race by finishing in the idle gap, which is why the symptom looked
    // scene-specific.
    //
    // Fix: triple-buffer. The tonemap writes into a *free* pool buffer; only
    // after that buffer's command buffer COMPLETES (signalled on a background
    // thread via `presentSync`) is it promoted to `outputTexture` on the next
    // tick. SceneKit thus always samples a fully-rendered buffer, and we never
    // pay a per-frame `waitUntilCompleted` stall. FOUR buffers (`ldrPoolCount`)
    // cover: the presented one, the one SceneKit may still be reading from its
    // previous composite, and the TWO that may be in flight at once now that
    // `maxFramesInFlight = 2` pipelines CPU encode against GPU execution.
    var ldrPool: [MTLTexture] = []
    var presentedIdx: Int = 0
    /// The buffer presented on the *previous* tick. SceneKit's compositor can
    /// lag the rebind by ~1 frame, so we must not write into it either.
    var prevPresentedIdx: Int = -1
    /// The pool buffer the current frame's tonemap writes into. Bound by
    /// `encodeTonemapPass`; chosen by `acquireWriteTarget()` each frame.
    var tonemapWriteTarget: MTLTexture!
    let presentSync = IlluminatoramaPresentSync()

    // Phase 2 — full-res HDR with SSR composited on top of direct lighting.
    // Bloom and tonemap read from this so reflections feed both effects.
    var hdrCompositeTexture: MTLTexture!

    /// Bloom mip pyramid (S1.2). SEPARATE non-mipmapped textures per level, not
    /// one mipmapped texture — three reasons, all of them Metal facts rather than
    /// taste: a compute `access::write` texture only ever addresses level 0, so
    /// mip writes would need a stored `makeTextureView` per level anyway;
    /// automatic hazard tracking is per-RESOURCE, so one mipmapped texture would
    /// serialise the whole chain against itself; and accumulating in place would
    /// want `access::read_write`, which for `rgba16Float` requires
    /// ReadWriteTextureTier2 (fine on Apple silicon, absent on older Intel Macs).
    ///
    /// `bloomDownChain[0]` is internal/2 and each level halves again;
    /// `bloomUpChain` is one SHORTER — the smallest level needs no accumulation
    /// buffer because it is read straight off the down chain to seed the up pass.
    /// `bloomUpChain[0]` (== internal/2) is what the tonemap samples.
    var bloomDownChain: [MTLTexture]!
    var bloomUpChain: [MTLTexture]!
    // Halation runs at HALF the internal resolution — the halo's tight orange core is a
    // few output pixels wide, which a quarter-res grid cannot carry.
    var halationBright: MTLTexture!
    var halationBlurH: MTLTexture!
    var halationBlurV: MTLTexture!

    // Issue #65 — 3D colour-grading LUT slot. A `type3D` cube sampled in the
    // tonemap pass AFTER all other post (display-space grade). Baked to identity
    // on init so the default is an EXACT no-op; a scene replaces it via
    // `setColorLUT(...)` (or `loadCubeLUT(...)`). `colorLUTAmount` (0 = off) blends
    // the graded result over the ungraded one. `colorLUTSize` is the per-axis
    // resolution used for the half-texel sampling correction.
    var colorLUT: MTLTexture!
    var colorLUTSize: Int = IlluminatoramaRenderer.colorLUTDefaultSize

    /// Last built-in look baked into `colorLUT`, so `applySharedColorGrade` only
    /// rebakes the (CPU-built) LUT texture when the chosen look actually changes.
    var appliedColorGradeLook: IlluminatoramaColorGradeLook = .none

    // Phase 2.7 — motion vectors + TAA.
    // Velocity is the 4th G-buffer color attachment; we ping-pong two HDR
    // history textures so the TAA pass can read from one and write the
    // resolved current frame into the other.
    var velocityTexture: MTLTexture!

    var historyA: MTLTexture!
    var historyB: MTLTexture!
    /// The TAA resolve's downstream output (see `Targets.taaResolved`).
    var taaResolvedTexture: MTLTexture!
    /// `taaStillCrossfade`'s frozen picture (DH-0952) — allocated on capture, released once the fade ends.
    var taaCrossfadeSource: MTLTexture?
    /// When true, historyA holds the previous frame's TAA output and we
    /// write the current frame's resolve into historyB. Toggled each frame.
    var historyToggle: Bool = false
    /// Previous frame's jittered view-projection matrix — drives the
    /// per-vertex prevClip computation in the G-buffer vertex shader.
    var previousViewProjection: simd_float4x4 = matrix_identity_float4x4
    /// The jittered VP computed this frame; copied into
    /// `previousViewProjection` at the end of render() so next frame's
    /// vertex shader sees the right matrix.
    var lastFrameViewProjection: simd_float4x4 = matrix_identity_float4x4
    /// The sub-pixel projection jitter baked into `previousViewProjection`, in
    /// NDC. Paired with `lastFrameJitterNDC` exactly as the two VP matrices are:
    /// the velocity buffer differences two JITTERED matrices, so removing the
    /// jitter again needs both frames' offsets, not just this one's.
    var previousJitterNDC: SIMD2<Float> = .zero
    /// The jitter applied this frame; copied into `previousJitterNDC` at the end
    /// of render() alongside the matrix it belongs to.
    var lastFrameJitterNDC: SIMD2<Float> = .zero
    /// Incremented every TAA-enabled render so the Halton(2,3) jitter walks
    /// through its sequence and accumulates as proper supersampling.
    var taaFrameIndex: UInt32 = 0
    /// True until the renderer has produced at least one valid history
    /// frame; the TAA kernel uses this to write current straight through.
    /// Reset on resize, on `taaEnabled` going false→true, and on instance
    /// buffer growth (which destroys previous-frame instance data).
    var taaNeedsFirstFrame: Bool = true
    /// Tracks the previous value of `taaEnabled` so a false→true transition
    /// re-primes the history.
    var previousTaaEnabled: Bool = false

    var taaResolvePipeline: MTLComputePipelineState!
    // Bloom mip pyramid (S1.2): prefilter (full-res HDR → mip0) + down + up.
    var bloomPrefilterPipeline: MTLComputePipelineState!
    var bloomDownPipeline: MTLComputePipelineState!
    var bloomUpPipeline: MTLComputePipelineState!
    var halationThresholdPipeline: MTLComputePipelineState!
    var halationBlurHPipeline: MTLComputePipelineState!
    var halationBlurVPipeline: MTLComputePipelineState!
    // Phase 4.28 — tonemap is now a fullscreen RENDER pass (vertex+fragment)
    // rather than a compute kernel, so its write to `outputTexture` is visible
    // to SceneKit's cross-queue background sample without a CPU wait. See the
    // shader comment on `illumi_tonemap_vs`.
    var tonemapPipeline: MTLRenderPipelineState!
    // Phase 4.21 — auto-exposure estimator. Runs once per frame before
    // bloom; reduces HDR luminance and EMAs into a tiny GPU-side state
    // buffer the tonemap then reads `exposure` from.
    var exposureEstimatePipeline: MTLComputePipelineState!
    /// 16-byte GPU-only `ExposureState` (prevTargetLogLum + smoothed
    /// exposure + new target + dt). Persistent across frames because
    /// the EMA needs its own history. Storage mode is `.shared` so the
    /// host can seed it once at init and inspect for diagnostics; the
    /// kernel writes back via a `device&` binding.
    var exposureBuffer: MTLBuffer!
    /// DH-0655 — `ExposureHistogram.bins` counts + total, p50, p95, p99, mean, key.
    var exposureHistogramBuffer: MTLBuffer!

    /// Cover-blit pipeline — fullscreen textured quad with UV crop, used by
    /// `present(to:)` to route the output texture into an MTKView drawable
    /// without a SceneKit pass. `nil` when the library doesn't include the
    /// shader (shouldn't happen; logged once at init).
    var coverBlitPipeline: MTLRenderPipelineState?

    /// True for the remainder of a frame once `encodePostResolveFX` has drawn
    /// the additive overlay (dust + sparks) into the display buffer. When set,
    /// downstream post-FX read `hdrTexture` (resolved opaque + overlay) instead
    /// of the bare `postFXSource`.
    var overlayApplied = false

    /// This frame's adapted HDR (nil = the pass didn't run). Reset before DOF each frame.
    var localToneMapOutput: MTLTexture?

    /// Phase 4.21 — estimate the scene's log-luminance once per frame and
    /// EMA-smooth into `exposureBuffer.smoothedExposure`. Lives between
    /// the TAA resolve and the bloom passes so it reads the post-TAA
    /// HDR composite (the same texture the tonemap will read) — the
    /// resulting exposure is consistent with what the tonemap then
    /// applies. Single threadgroup, 256 threads, shared-mem reduction.
    /// Wall-clock of the previous `encodeExposureEstimate` call. Used to
    /// derive a real per-frame `dt` for the EMA so the smoothing keeps
    /// the same half-life regardless of the host's frame rate.
    var lastExposureTickTime: CFTimeInterval = CACurrentMediaTime()
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// TAA, bloom, halation, tonemap and exposure-estimate pipelines.
    @MainActor static func buildPostPipelines(into s: PostFXState, cache: SimPipelineCache, device: MTLDevice, library: MTLLibrary) throws {
        guard let taaResolvePipelineBuilt = cache.pipelineState(name: "illumi_taa_resolve", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_taa_resolve")
        }
        s.taaResolvePipeline = taaResolvePipelineBuilt
        guard let bloomPrefilterPipelineBuilt = cache.pipelineState(name: "illumi_bloom_prefilter", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_bloom_prefilter")
        }
        s.bloomPrefilterPipeline = bloomPrefilterPipelineBuilt
        guard let bloomDownPipelineBuilt = cache.pipelineState(name: "illumi_bloom_down", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_bloom_down")
        }
        s.bloomDownPipeline = bloomDownPipelineBuilt
        guard let bloomUpPipelineBuilt = cache.pipelineState(name: "illumi_bloom_up", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_bloom_up")
        }
        s.bloomUpPipeline = bloomUpPipelineBuilt
        guard let halationThresholdPipelineBuilt = cache.pipelineState(name: "illumi_halation_threshold", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_halation_threshold")
        }
        s.halationThresholdPipeline = halationThresholdPipelineBuilt
        guard let halationBlurHPipelineBuilt = cache.pipelineState(name: "illumi_halation_blur_h", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_halation_blur_h")
        }
        s.halationBlurHPipeline = halationBlurHPipelineBuilt
        guard let halationBlurVPipelineBuilt = cache.pipelineState(name: "illumi_halation_blur_v", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_halation_blur_v")
        }
        s.halationBlurVPipeline = halationBlurVPipelineBuilt
        // Phase 4.28 — tonemap render pipeline (fullscreen triangle → fragment).
        // Outputs into the bgra8Unorm `outputTexture` via a render pass so the
        // write is visible to SceneKit's cross-queue background sample.
        let tmDesc = MTLRenderPipelineDescriptor()
        tmDesc.label = "Illuminatorama.tonemap"
        tmDesc.vertexFunction = library.makeFunction(name: "illumi_tonemap_vs")
        tmDesc.fragmentFunction = library.makeFunction(name: "illumi_tonemap_fs")
        tmDesc.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        let tonemap = try device.makeRenderPipelineState(descriptor: tmDesc)
        // Phase 4.21 — auto-exposure estimator.
        s.tonemapPipeline = tonemap
        guard let exposureEstimatePipelineBuilt = cache.pipelineState(name: "illumi_exposure_estimate", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_exposure_estimate")
        }
        s.exposureEstimatePipeline = exposureEstimatePipelineBuilt
    }

    /// Exposure state + histogram buffers (seeded for the first frame).
    @MainActor static func buildExposureBuffers(into s: PostFXState, device: MTLDevice) throws {
        // ExposureState is 16 bytes (4 floats). Storage `.shared` so
        // the renderer can seed it once and (if needed) read it for
        // diagnostics. The kernel updates it in place each frame.
        guard let expoBuf = device.makeBuffer(
            length: 16, options: .storageModeShared
        ) else { throw IlluminatoramaError.bufferAllocationFailed("exposureState") }
        expoBuf.label = "Illuminatorama.exposureState"
        // Initial state: previous-frame target log lum = -2.0 (mid-grey-ish);
        // smoothed exposure = 1.0 (the static fallback); new target same;
        // dt = 1/60 s (host will overwrite per frame).
        let initialState: [Float] = [-2.0, 1.0, -2.0, 1.0 / 60.0]
        memcpy(expoBuf.contents(), initialState,
               MemoryLayout<Float>.stride * initialState.count)
        s.exposureBuffer = expoBuf
        guard let histBuf = device.makeBuffer(length: MemoryLayout<Float>.stride * (ExposureHistogram.bins + 8),
                                              options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("exposureHistogram")
        }
        histBuf.label = "Illuminatorama.exposureHistogram"
        memset(histBuf.contents(), 0, histBuf.length)
        s.exposureHistogramBuffer = histBuf
    }

    /// The identity 3D colour-grade LUT (a no-op until a scene calls `setColorLUT`).
    @MainActor static func buildIdentityColorLUT(into s: PostFXState, device: MTLDevice) throws {
        guard let lut = Self.makeIdentityColorLUT(device: device,
                                                  size: Self.colorLUTDefaultSize) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.colorLUT")
        }
        s.colorLUT = lut
    }
}
