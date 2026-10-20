import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Present the current `outputTexture` directly into an `MTKView`
    /// drawable, bypassing SceneKit entirely. Applies a cover-fill UV crop
    /// (same math as the SceneKit `contentsTransform` path) so the fixed
    /// render canvas fills the drawable without stretching or letterboxing.
    ///
    /// Call this from the scene's tick loop INSTEAD of assigning
    /// `scene.background.contents = outputTexture`. The two paths are
    /// mutually exclusive — calling both wastes a full GPU pass.
    ///
    /// Safe to call when `drawable` is nil (no-ops). The blit is
    /// fire-and-forget on the renderer's own command queue; no CPU wait.
    public func present(to drawable: (any CAMetalDrawable)?) {
        guard let drawable,
              let pipeline = pfx.coverBlitPipeline,
              let cb = commandQueue.makeCommandBuffer() else { return }
        cb.label = "Illuminatorama.coverBlit"
        let w = drawable.texture.width
        let h = drawable.texture.height
        // A zero-size drawable means the MTKView has no window yet (tick loop
        // fires before MetalViewRenderer.makeNSView). Presenting it produces a
        // nil `priv` in the system's IOSurface callback → EXC_BAD_ACCESS in
        // layer_private_present_impl. Skip until the view is live.
        guard w > 0, h > 0 else { return }
        var uvRect = coverUVRect(viewportWidth: w, viewportHeight: h)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture    = drawable.texture
        pass.colorAttachments[0].loadAction  = .dontCare
        pass.colorAttachments[0].storeAction = .store

        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentTexture(outputTexture, index: 0)
        enc.setFragmentBytes(&uvRect,
                             length: MemoryLayout<SIMD4<Float>>.stride,
                             index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()

        // Feed the recorder (if any). Fired after the present commit so the
        // tap's copy enqueues behind it on the same queue — the texture is
        // fully written by then.
        onFramePresented?(outputTexture, commandQueue)
    }

    /// Test / determinism hook — force every temporal accumulator and the
    /// auto-exposure EMA back to a cold-start state on the **next** frame,
    /// exactly as a fresh renderer would begin.
    ///
    /// Why this exists: `IlluminatoramaRenderer(engine: .shared)` is a
    /// process-global engine. Its TAA history, SSAO/SSR/RT-GI temporal
    /// accumulators, post-FX easing state, and auto-exposure EMA all persist
    /// across renderer instances, so absolute-threshold GPU tests that build a
    /// fresh bridge can flake by suite position (the previous test's converged
    /// history bleeds into the next test's first frame). Calling this between
    /// tests makes each test start from the same cold history, restoring
    /// determinism.
    ///
    /// It is **purely additive**: it does nothing to normal per-frame rendering
    /// unless explicitly invoked, so production callers (e.g. Visualizer) are
    /// unaffected. Under the hood it composes the *same* reset paths the engine
    /// already runs on a resize (the four `…NeedsFirstFrame` flags, which make
    /// the next frame reproject against nothing), re-seeds the GPU
    /// `ExposureState` buffer to its constructor cold-start values, and clears
    /// the host-side post-FX easing / exposure-tick timers so the next frame
    /// snaps to targets rather than gliding from stale ones.
    public func resetTemporalHistory() {
        // Temporal reprojection passes: reproject against nothing next frame,
        // identical to the post-resize reset above.
        pfx.taaNeedsFirstFrame  = true
        aos.aoNeedsFirstFrame   = true
        ssrs.ssrNeedsFirstFrame  = true
        rtGINeedsFirstFrame = true
        aos.aoAccumulator?.reset()

        // Auto-exposure EMA — re-seed the GPU ExposureState buffer to the exact
        // cold-start values written once in the constructor:
        //   [prevTargetLogLum = -2.0, smoothedExposure = 1.0,
        //    newTargetLogLum = -2.0, deltaTime = 1/60].
        let initialExposureState: [Float] = [-2.0, 1.0, -2.0, 1.0 / 60.0]
        memcpy(pfx.exposureBuffer.contents(), initialExposureState,
               MemoryLayout<Float>.stride * initialExposureState.count)

        // Host-side post-FX easing: clear the tick timer so the next
        // `advancePostFXEasing` sees dt == 0 → k == 1 and snaps every eased
        // value to its current target (no startup glide from stale eased state).
        pfx.lastPostFXEaseTime   = 0
        // Re-seed the self-timed exposure/particle tick clocks to now so the
        // first post-reset frame reports a sane dt rather than a huge gap.
        pfx.lastExposureTickTime = CACurrentMediaTime()
        fxs.lastParticleTickTime = CACurrentMediaTime()
    }

    // ── Presented-buffer pool helpers ─────────────────────────────────────────

    // 4 = presented + previously-presented (both still readable by consumers) +
    // the two frames that may be in flight at once (`maxFramesInFlight = 2`).
    static let ldrPoolCount = 4

    /// Allocate one LDR (`bgra8Unorm_srgb`) output texture for the present pool.
    /// sRGB format: the tonemap writes LINEAR and the GPU store applies the sRGB
    /// OETF. SceneKit samples this as `background.contents`, decodes it back to
    /// linear, then re-encodes for its drawable — one correct round-trip. (A
    /// non-sRGB format + a manual `pow(1/2.2)` in the shader double-encoded
    /// against SceneKit's pass and washed the colours out.) sRGB targets don't
    /// support `.shaderWrite`; the tonemap is a render pass, so we don't need it.
    /// Clear freshly allocated presentable textures to opaque black, on this renderer's own queue
    /// (so the clear is ordered ahead of any later `present` / render of them).
    ///
    /// A new `.private` texture's contents are undefined, and on macOS they read back as solid
    /// 255,0,255 — measured 2026-10-15: every sampled texel of six freshly made 1600×1000
    /// `bgra8Unorm_srgb` targets, with and without API validation. Anything that can reach the
    /// screen before a frame writes it (the first present after init; a slot promoted early) must
    /// therefore be initialised, or it shows as a fuchsia flash.
    static func clearLDRTextures(_ textures: [MTLTexture], on queue: MTLCommandQueue) {
        guard !textures.isEmpty, let cb = queue.makeCommandBuffer() else { return }
        cb.label = "Illuminatorama.ldr.clear"
        for tex in textures {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = tex
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            cb.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
        }
        cb.commit()
    }

    static func makeLDRTexture(device: MTLDevice, w: Int, h: Int,
                                        label: String) -> MTLTexture? {
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        d.pixelFormat = .bgra8Unorm_srgb
        d.width = max(1, w)
        d.height = max(1, h)
        d.usage = [.shaderRead, .renderTarget]
        d.storageMode = .private
        let t = device.makeTexture(descriptor: d)
        t?.label = label
        return t
    }

    /// Promote the most-recently-completed pool buffer to `outputTexture`.
    /// Called at the top of each `render()` (and after the snapshot barrier)
    /// so consumers always bind a fully-rendered buffer. No-op until a buffer
    /// finishes. Internal (not private) so `IlluminatoramaOverlay.renderForSnapshot`
    /// can promote the just-rendered frame after its completion barrier.
    func promoteCompletedBuffer() {
        guard let idx = pfx.presentSync.takeCompleted(),
              idx >= 0, idx < pfx.ldrPool.count, idx != pfx.presentedIdx else { return }
        pfx.prevPresentedIdx = pfx.presentedIdx
        pfx.presentedIdx = idx
        outputTexture = pfx.ldrPool[idx]
        if Self.swapProbeEnabled { recordSwapInterval() }   // issue #65 jitter probe
    }

    /// Choose a pool index to render into this frame: anything that is neither
    /// the currently presented buffer, NOR the previously presented buffer
    /// (SceneKit / the cover blit may still be sampling either), NOR a buffer
    /// whose own render is STILL IN FLIGHT. The in-flight exclusion matters now
    /// that `maxFramesInFlight = 2`: two consecutive `render()` calls can both be
    /// queued before either completes, and they must target DIFFERENT buffers or
    /// the second would clobber the first before it is ever presented. With the
    /// 4-buffer pool the worst case is presented + prev + one already in flight =
    /// 3 excluded, leaving exactly one free slot for this frame.
    func acquireWriteTarget() -> Int {
        for i in 0..<pfx.ldrPool.count
        where i != pfx.presentedIdx && i != pfx.prevPresentedIdx && !pfx.presentSync.isInFlight(i) {
            return i
        }
        // Degenerate fallback (pool too small or all excluded): avoid the presented.
        return (pfx.presentedIdx + 1) % max(1, pfx.ldrPool.count)
    }
}

// MARK: - Frame / init steps extracted from the main file

extension IlluminatoramaRenderer {
    /// Install the completion handler (present-sync, GPU meter, pass timer, adaptive latency guard, awaited-frame signal) and commit. `addCompletedHandler` is only legal before `commit()`.
    func commitFrame(_ cb: MTLCommandBuffer, writeIdx: Int, awaited: Bool) {
        let completedIdx = writeIdx
        let completedGeneration = pfx.presentSync.currentGeneration
        let raceSlot = Self.raceProbeEnabled ? lts.frameRingIndex : -1
        let sync = pfx.presentSync
        let sem = gbs.inFlightSemaphore
        let meter = gbs.gpuMeter
        let pt = gbs.passTimer!
        let errPath = Self.frameEnv("VIZ_ILLUMI_CBERROR_PATH")
        // The awaited-frame signal, if `renderAwaited()` armed one. It must be created and
        // installed HERE — `addCompletedHandler` is only legal before `commit()`, so a caller
        // cannot decide to wait asynchronously after the fact.
        let completion: FrameCompletionSignal? = awaited ? FrameCompletionSignal() : nil
        gbs.pendingFrameCompletion = completion
        cb.addCompletedHandler { buf in
            meter.record(buf)
            let gpuMs = (buf.gpuEndTime - buf.gpuStartTime) * 1000.0
            pt.resolve(frameGpuMs: gpuMs)
            // Diagnostic (sandbox eats os_log): on a GPU fault, write the error +
            // status to a sidecar file so the headless harness can read WHY the
            // frame faulted (vs the swallowed magenta). No-op without the env var.
            if buf.status == .error || buf.error != nil, let p = errPath {
                let nsErr = buf.error as NSError?
                let line = "frame status=\(buf.status.rawValue) domain=\(nsErr?.domain ?? "?") code=\(nsErr?.code ?? 0) \(nsErr?.localizedDescription ?? "")\n"
                if let data = line.data(using: .utf8) {
                    if let h = FileHandle(forWritingAtPath: p) {
                        h.seekToEndOfFile(); h.write(data); try? h.close()
                    } else { try? data.write(to: URL(fileURLWithPath: p)) }
                }
            }
            sync.markCompleted(completedIdx, generation: completedGeneration)
            if raceSlot >= 0 { sync.raceProbeCompleteFrame(slot: raceSlot) }
            // Adaptive latency guard (replaces a plain `sem.signal()`): update the
            // GPU-time EMA and release THIS frame's permit — UNLESS frames are heavy
            // enough that a 2nd pipelined frame only adds ~one frame of input latency
            // without raising throughput (GPU frame >> tick). In that case HOLD the
            // permit, shrinking the effective in-flight depth from 2 to 1 so heavy RT
            // scenes stay responsive. Converges back to 2 when frames get light again.
            sync.frameCompletedAdaptive(gpuMs: gpuMs, semaphore: sem)
            // LAST, and after `markCompleted` above: an awaiter resumes into
            // `promoteCompletedBuffer()`, which can only find this frame once `writeIdx` is
            // published. Same ordering `waitUntilCompleted()` gives the blocking path.
            completion?.signal()
        }
        gbs.lastCommittedFrameCB = cb
        cb.commit()
    }
}
