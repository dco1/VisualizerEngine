import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    // ── Render ────────────────────────────────────────────────────────────────

    /// - Parameter blocking: the CAPTURE contract — `true` is the headless snapshot /
    ///   export path, `false` the interactive tick.
    ///
    ///   `false` pipelines. A back-pressured GPU DROPS the tick rather than stalling the
    ///   main thread, and the frame this call commits is not the one `outputTexture`
    ///   holds on return — the pool promotes it on the NEXT tick. That is correct for a
    ///   display loop, which only ever wants the freshest fully-rendered buffer.
    ///
    ///   `true` (a) waits for an in-flight slot instead of dropping, because a dropped
    ///   export render leaves `outputTexture` on a stale / never-rendered pool buffer →
    ///   captured as magenta (no `cb.error`); and (b) waits for THIS frame on the GPU and
    ///   promotes it before returning, so `outputTexture` IS the frame you just asked for.
    ///   Without (b) a readback taken straight after a scene mutation samples the buffer
    ///   promoted at the TOP of this call — a frame rendered before the mutation — which
    ///   reads exactly like "the instance upload landed a frame late". It doesn't:
    ///   `uploadInstances` runs the same frame the change is seen. The latency was always
    ///   the presented-buffer pool, never the upload.
    ///
    /// Returns `true` if this call actually produced a frame, `false` if it
    /// dropped (GPU still draining the previous frames — the non-blocking
    /// `inFlightSemaphore` path) or bailed on a transient failure. Callers
    /// use the result to count FPS against *rendered* frames rather than the
    /// fixed Timer cadence, and to skip re-presenting a stale `outputTexture`.
    @discardableResult
    public func render(blocking: Bool = false) -> Bool {
        // Read-and-CLEAR the awaited-frame arming here, before any early return, so no exit
        // path can leave it set for the next caller (see `renderAwaited()`). A nil
        // `pendingFrameCompletion` is what an early return leaves behind, and
        // `awaitCommittedFrame()` reads that as "nothing to wait for".
        let awaited = gbs.armAwaitedFrame
        gbs.armAwaitedFrame = false
        if awaited { gbs.pendingFrameCompletion = nil }
        // Fail LOUD on a boot-relative animation clock. `time` is a Float used as a phase by
        // every GPU oscillator, so a large absolute value quantises all of them (see `time`'s
        // doc and `RenderClock`). This is the check that would have caught the defect on the
        // first frame instead of after it silently killed the film grain; it costs one compare
        // per frame and never fires for a host that sends a session-relative clock.
        if !pfx.warnedCoarseAnimationClock && RenderClock.isTooCoarse(CFTimeInterval(time)) {
            pfx.warnedCoarseAnimationClock = true
            let m = RenderClock.coarseClockDiagnosis(CFTimeInterval(time)) + "\n"
            FileHandle.standardError.write(Data(m.utf8))
        }
        // Promote any pool buffer whose render completed since last tick to
        // the presented `outputTexture`. Done first so consumers binding
        // `outputTexture` this tick see the freshest fully-rendered frame.
        promoteCompletedBuffer()
        // A scene with NO opaque instances is still renderable when it has glass
        // panes (Floating Lenses): the deferred pipeline runs with an empty
        // G-buffer (every pixel resolves to sky in `illumi_lighting`), then the
        // forward glass pass draws the lenses over that sky. Only a truly empty
        // frame (no opaque AND no glass) falls back to the flat sky clear.
        guard !instances.isEmpty || !glassInstances.isEmpty || !glassMeshGroups.isEmpty else {
            // Nothing to draw — still clear the output to the sky color so the
            // host's display quad isn't showing last frame's garbage. This IS a
            // produced frame (the sky), so it counts toward FPS.
            blankSky()
            return true
        }
        // Throttle: drop this tick if the GPU hasn't drained the previous
        // frame yet (non-blocking wait). Prevents queue saturation from
        // freezing the main thread / stalling presentation on heavy scenes.
        // Balanced by `inFlightSemaphore.signal()` in the completion handler.
        // The snapshot export path waits instead (blocking) so it never drops.
        if blocking {
            gbs.inFlightSemaphore.wait()
        } else {
            // Dropped frame — GPU hasn't drained the previous frames. Report
            // `false` so the caller doesn't count it as a rendered frame.
            guard gbs.inFlightSemaphore.wait(timeout: .now()) == .success else { return false }
        }
        // ── Panel is the single source of truth for TAA on/off ──────────────
        // Authority lives HERE, once, instead of in every scene controller. The
        // panel toggle (IlluminatoramaSharedSettings.shared.taaEnabled) wins for
        // every Illuminatorama scene unless that scene explicitly opts out via
        // `sharedTAAOverride`. This is why a scene's own `renderer.taaEnabled = …`
        // write no longer leaks the toggle out of sync — it's overwritten here
        // each frame before any TAA gating reads it.
        taaEnabled = Self.taaEnvOverride ?? sharedTAAOverride ?? IlluminatoramaSharedSettings.shared.taaEnabled
        // Same authority model for the SVGF GI denoiser: the panel toggle drives it
        // for every Illuminatorama scene. (`effectiveSVGFEnabled` still layers the
        // VIZ_ILLUMI_SVGF env A/B override on top.) Safe to apply universally — the
        // cascade self-gates on `rtGITemporalEnabled`, so scenes that force temporal
        // off (CoinPusher, HotdogPress) stay on the bilateral path regardless.
        svgfEnabled = IlluminatoramaSharedSettings.shared.svgfEnabled
        primeTemporalHistoriesOnTransitions()
        // ── Panel is the single source of truth for lens aberration too ─────
        // CA / spherical aberration / fringe (+ post-FX easing) have NO per-scene
        // overrides anywhere, so apply them HERE once for every Illuminatorama
        // scene — including the ones that never wired post-FX and every future
        // scene, which inherit them with zero per-scene code. (Bloom is NOT here:
        // scenes art-direct it and `render()` runs after their tick, so it opts in
        // via `applySharedPostFX`.) Targets are set before `uploadFrameUniforms`
        // below, which eases toward them.
        if appliesSharedLensFX { applySharedLensFX(); applySharedColorGrade() }
        if appliesSharedDenoiser { applySharedDenoiser() }
        // Ping-pong the instance buffer FIRST: after this swap,
        // currentInstanceBuffer is what was previousInstanceBuffer last frame
        // (so uploadInstances overwrites it with this frame's data), and
        // previousInstanceBuffer now points to last frame's data.
        gbs.useBufferA.toggle()
        // Advance the per-frame input-buffer ring in lockstep. From here on,
        // `frameUniformBuffer` / the light buffers / `superquadricParamBuffer`
        // resolve to THIS frame's slot — distinct from the slot the previously
        // pipelined frame (still on the GPU) is reading. Done before the uploads
        // below write into the new slot, and before any encode binds it.
        lts.frameRingIndex = (lts.frameRingIndex + 1) % lts.frameUniformRing.count
        if Self.raceProbeEnabled {
            if Self.raceProbePin0 { lts.frameRingIndex = 0 }   // negative control: pre-fix single-buffer
            pfx.presentSync.raceProbeBeginFrame(slot: lts.frameRingIndex)
        }
        ensureBufferCapacity()  // may set taaNeedsFirstFrame on growth
        updateSceneStability()
        updateCascades()
        // Phase 4.10 — compute per-spot shadow matrices + slice indices
        // BEFORE `uploadSpotLights` so the GPU buffer sees the current
        // frame's values. Done unconditionally even when shadows are
        // disabled (cheap CPU math); the shadow PASS gates on
        // `spotShadowsEnabled` to skip the expensive depth render.
        updateSpotShadows()
        // Point-light cube shadows: assign cube pages + compute the 6 face VPs.
        // No-op (clears indices to -1) when the feature is off, so the uploaded
        // buffer never carries a stale page. Cheap CPU math; the depth PASS gates.
        updatePointShadows()
        uploadFrameUniforms()
        uploadInstances()
        computeCullBounds()   // DH-0534 — reads the CPU-written instance copy, before any GPU hook
        uploadPointLights()
        uploadSpotLights()
        uploadAreaLights()
        uploadExtraDirectionals()
        uploadInteriorApertures()

        // Pick a free pool buffer for this frame's tonemap output (never the
        // presented one SceneKit may sample, nor the one still in flight).
        let writeIdx = acquireWriteTarget()
        pfx.tonemapWriteTarget = pfx.ldrPool[writeIdx]

        guard let cb = commandQueue.makeCommandBuffer() else {
            Self.log.error("makeCommandBuffer() returned nil")
            gbs.inFlightSemaphore.signal()  // release the slot acquired above
            return false
        }
        // Reserve this pool slot until the GPU finishes writing it, so a second
        // pipelined frame (maxFramesInFlight = 2) picks a different buffer. Cleared
        // by `markCompleted` in the completion handler below. Marked after the
        // makeCommandBuffer guard so a failed allocation doesn't leak the slot.
        pfx.presentSync.markInFlight(writeIdx)
        cb.label = "Illuminatorama.frame"
        gbs.passTimer.beginFrame()   // reset per-pass GPU-timer sample assignment (env-gated)
        // DH-1473 — no pass may trace a TLAS that `updateRTAccel` will rebuild later in this frame.
        retireTLASIfTopologyMoved()
        // Opt-in host GPU work for THIS frame (see `onEncodeFrameCompute`); nil → nothing.
        onEncodeFrameCompute?(cb)

        // Opt-in GPU-resident instance write (additive — see `onEncodeGPUInstances`).
        // `uploadInstances()` above already reserved the slots + built `meshGroups`;
        // here, before any pass reads the instance buffer, a scene kernel can
        // overwrite its group's slots with GPU-computed data. No-op (closure nil)
        // for every scene that doesn't set the hook.
        if let hook = onEncodeGPUInstances {
            let ranges = gbs.meshGroupRange
            hook(cb, currentInstanceBuffer) { kind in ranges[kind] }
        }
        // GPU-resident point lights (additive — see `onEncodeGPUPointLights`).
        // `uploadPointLights()` above packed the CPU placeholders into the buffer;
        // the scene kernel overwrites the reserved slots before the lighting pass.
        if let plHook = onEncodeGPUPointLights, !pointLights.isEmpty {
            plHook(cb, pointLightBuffer, pointLights.count)
        }

        // Phase 4.11 — tick particle simulations BEFORE rendering them.
        // The integration is per-frame; the additive draw lands later in
        // the frame graph (after SSR composite but before TAA history /
        // bloom / tonemap so the additive accumulation participates in
        // those passes naturally).
        encodeParticleStep(cb)

        // Phase 4.13a — repack any compute-fed `DynamicMesh` geometry
        // from separate pos/norm into Illuminatorama's interleaved
        // vertex layout. Must run before shadow / G-buffer reads.
        encodeGPURepacks(cb)

        // Issue #65 — on a GPU mesh's FIRST frame, seed prev = current so its
        // motion vector starts at zero (no first-frame streak). Subsequent frames
        // are seeded by the post-G-buffer capture below.
        encodeGPUMeshPrevCapture(cb, prime: true)

        encodeShadowPasses(cb)
        encodeSpotShadowPasses(cb)
        encodePointShadowPasses(cb)
        encodeGBufferPass(cb)
        onEncodeAfterGBuffer?(cb)   // DH-1466 — nil for every host that does not opt in
        // Issue #65 — now that the G-buffer draw has read THIS frame's prev
        // positions, copy the current positions forward to be next frame's prev.
        encodeGPUMeshPrevCapture(cb, prime: false)
        encodeIBLBakeIfNeeded(cb)
        // Phase 4.39: SSAO → bilateral spatial → temporal → lighting (reads denoised AO).
        encodeSSAOPass(cb)
        encodeSSAOSpatialFilter(cb)
        encodeSSAOTemporalPass(cb)
        prepareAOVFrame()   // DH-1304 — this frame's capture (no-op unless `aovCaptureEnabled`)
        encodeCloudShadowPass(cb)   // DH-1462 — no-op unless `cloudShadowEnabled` (and the sky changed)
        encodeLightingPass(cb)
        // Issue #65 — screen-space SSS: diffuse the lit irradiance of SSS-flagged
        // pixels and blend it back into hdrTexture BEFORE SSR/RT/TAA/bloom read it,
        // so the soft light-bleed propagates through the rest of the pipeline. No-op
        // (and not encoded) unless a scene set sssStrength > 0.
        encodeSSSPasses(cb)
        // Phase 4.43 — additive particle / external-point draws (emitter
        // sparks + dust motes) MOVED to AFTER the TAA resolve (see
        // `encodePostResolveFX`). They have no motion vectors of their own,
        // so flowing them through TAA history reprojected each moving mote as
        // static and retained ~90% of the previous frame → visible trails.
        // Trade-off vs the old Phase 4.42 placement: these FX no longer feed
        // the SSR/RT reflection chain (they reflected in nearby metals). For
        // dust/sparks that's an acceptable loss; trail-free wins.
        // Phase 4.39: SSR gather → temporal → composite into hdrComposite.
        // PERF: with SSR off (`ssrIntensity <= 0` — Daydream's default config, and
        // the documented setting for RT-reflection hosts) the ray-march gather, the
        // temporal denoise, and the full-res composite are all pure waste — the
        // shader would early-out per pixel but the dispatches, depth reads, and the
        // hdr→hdrComposite copy still ran. Skip all three and seed `hdrComposite`
        // with one blit instead (downstream additive passes — RT sun/GI, glass
        // backdrop, bloom/tonemap, TAA input — all read `hdrCompositeTexture`).
        if ssrIntensity > 0 {
            encodeSSRGather(cb)
            encodeSSRTemporalPass(cb)
            encodeSSRComposite(cb)
        } else {
            encodeSSRBypassCopy(cb)
        }
        // Incremental invalidation (default-on): re-frame moved instances' cards
        // in place + flag them dirty so the update below re-lights only them,
        // preserving stationary cards' accumulation. Rigid movers AND deforming
        // meshes run as GPU kernels on this command buffer (#60 item 1 + Phase D);
        // the VIZ_SURFCACHE_GPU_DIFF=0 kill-switch takes the CPU loop. No-op
        // unless on.
        applyIncrementalSurfaceCacheUpdate(cb)
        // Curve primitives on the SOUP path (#60 item 7 incr. 2): adopt registry
        // changes (rebuild the soup AS with the curve geometry — topology), then
        // re-displace the control points for wind sway + refit the AS on this cb,
        // BEFORE the cache update + RT pass trace it. No-op for TLAS-path renderers
        // (guarded inside) and curve-free soup scenes.
        syncSoupCurveRegistry()
        encodeCurveWindDisplaceAndRefit(cb)
        // Phase 5 / A1 — dynamic residency: reassign atlas slots to this frame's
        // working set (from the previous frame's feedback) BEFORE the update relights
        // them. After the re-frame above so a promoted card's dirty flag survives.
        // No-op unless streaming is on with a budget.
        runStreamingResidency()
        // Re-light the on-surface radiance atlas (ping-pong) so the RT pass can
        // read cached multi-bounce radiance at its hits. No-op unless enabled.
        encodeSurfaceCacheUpdate(cb)
        // Phase 5 / B1 — spatially denoise the freshly-updated atlas into the
        // consumer atlas BEFORE the RT lighting pass reads it. No-op unless
        // `surfaceCacheDenoise` is on.
        encodeSurfaceCacheDenoise(cb)
        // Instanced-RT for extracted scenes: build/refit the TLAS this frame.
        // Sets `rtTLASActive` → trace the instance AS; otherwise fall back to
        // the primitive-AS soup / room path. No-op unless extraction RT is on.
        updateRTAccel(cb)
        // Hardware RT sun lighting (soft shadows + 1-bounce GI), added into the
        // HDR composite before particles / TAA / bloom. No-op unless enabled.
        //
        // #60 perf — gate the TLAS lighting pass on `rtEnabled`, not just on a live
        // TLAS. AAA glass builds the TLAS for refraction even when a scene only
        // wants glass (not RT GI/shadows on its opaque geometry); without this gate
        // such a scene paid the full RT-opaque-lighting pass (~25–50 ms at the lab's
        // ray counts) for nothing. The opaque floor then renders with the cheaper
        // deferred lighting; glass still refracts/reflects it via the TLAS. The soup
        // `else` branch already self-gates on `rtEnabled`.
        //
        // `rtOpaqueLightingEnabled` (default true) is a SECOND gate so a scene that
        // needs the surface cache / caustics (which require `rtEnabled`) can still
        // skip THIS opaque pass without losing the cache. With it false here, the
        // opaque geometry falls back to deferred lighting (exactly as in the
        // `rtEnabled == false` case), the surface-cache update + caustic splat still
        // run, and the glass still refracts via the live TLAS. See the flag's doc.
        //
        // C2 — the denoise chain is now shared. It used to live INSIDE the soup
        // `else` branch, so the TLAS path composited 4 GI rays raw; in an app that
        // also runs with TAA off, nothing cleaned them at all ("dark noisy
        // blotches"). Both lighting kernels now write the same `rtDiffuseTexture`,
        // so the same chain reads it whichever one ran.
        if rts.rtTLASActive && rtEnabled && rtOpaqueLightingEnabled {
            encodeRTLightingTLASPass(cb)
            encodeRTDiffuseDenoiseChain(cb)
        } else if !rts.rtTLASActive {
            encodeRTLightingPass(cb)
            encodeRTDiffuseDenoiseChain(cb)
        }
        // Volumetric god-ray shaft in the air. No-op unless enabled.
        encodeVolumetricPass(cb)
        encodeVolumetricRTPass(cb)
        encodeSpotBeamPass(cb)
        // In-view perspective cloud composite (issue #61). No-op unless a scene
        // opted in. Lands before TAA (so clouds anti-alias) and before the
        // additive firework particles in encodePostResolveFX (so clouds sit
        // behind them).
        encodeCloudPass(cb)
        // Glass caustics (#60): photon-trace sun→glass→receiver, splat into the
        // surface-cache cards, add per primary pixel. After RT lighting (HDR lit)
        // and before the glass draw. No-op unless caustics + cache + glass are on.
        encodeCausticPass(cb)
        // Forward reflective glass pane, blended over the lit composite (before
        // TAA so its reflections bloom/tonemap). No-op unless a glass pane is set.
        encodeGlassPass(cb)
        encodeTAAResolve(cb)
        // Stash this frame's depth as next frame's "previous" for the resolve's
        // disocclusion test. The resolve above already consumed last frame's
        // copy. The in-place copy is a GPU blit on this command buffer, and buffers on
        // one queue execute in order, so it lands before the next frame's G-buffer pass
        // overwrites `depthTexture` whatever the in-flight pool depth. Skipped when TAA is off (nothing reads it); the
        // enable-transition's taaNeedsFirstFrame covers the first re-enabled frame.
        if taaEnabled, let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: gbs.depthTexture, to: gbs.previousDepthTexture)
            blit.endEncoding()
        }
        // Phase 4.43 — additive overlay FX (dust + emitter sparks) composited
        // AFTER the resolve, into a throwaway display buffer, so the temporal
        // history never sees them and they can't trail. No-op when there are
        // no emitters/fields.
        encodePostResolveFX(cb)
        // Phase 4.21 — auto-exposure estimate. Runs AFTER TAA resolve so
        // we sample the same texture the tonemap will read, and BEFORE
        // bloom + tonemap so the smoothed exposure is fresh by the time
        // the tonemap binds the exposure buffer.
        encodeExposureEstimate(cb)
        pfx.localToneMapOutput = nil
        // Depth of field on the resolved HDR before bloom, so out-of-focus
        // bright dust motes bloom into soft bokeh. No-op unless enabled.
        encodeDOFPass(cb)
        // Feathered halo glow around selected objects. After DOF (outline stays
        // sharp even on out-of-focus geometry) and before bloom (halo blooms soft).
        encodeSelectionOutlinePass(cb)
        // Local adaptation (opt-in) of the finished optical image; bloom / halation /
        // tonemap read its output through `bloomTonemapSource`.
        encodeLocalToneMap(cb)
        encodeBloomPasses(cb)
        // Film halation reads the same HDR source bloom does (post-TAA, post-DOF), so
        // a blown out-of-focus highlight halates as softly as it blooms. No-op — not
        // even encoded — unless halationIntensity > 0.
        encodeHalationPasses(cb)
        encodeTonemapPass(cb)

        // Signal — on GPU completion (background thread) — that this pool
        // buffer is fully rendered and safe to present. `promoteCompletedBuffer`
        // (next tick, main actor) reads this and swaps `outputTexture`. Capturing
        // a plain `Int` keeps the handler `Sendable`-clean.
        commitFrame(cb, writeIdx: writeIdx, awaited: awaited)

        // CAPTURE CONTRACT (see `blocking` on this method). A snapshot caller must be
        // able to render and then immediately read `outputTexture` and get THIS frame.
        // The pool never promotes the in-flight buffer — `promoteCompletedBuffer()` at
        // the top of the NEXT tick does — so without this wait every capture reads a
        // frame at least one behind, and any test that mutates the scene and renders
        // once is asserting against the pre-mutation image. `waitUntilCompleted()`
        // returns only after the completion handler above has run, so `markCompleted`
        // has already published `writeIdx` and the promote below always finds it.
        // Live rendering (`blocking == false`) is untouched and stays pipelined.
        // `awaited` keeps the never-drop half of the contract (the semaphore wait above) and
        // moves ONLY the CPU stall: `awaitCommittedFrame()` performs the same wait-then-promote
        // from a suspension instead of from `__psynch_cvwait` on the caller's thread.
        if blocking && !awaited {
            cb.waitUntilCompleted()
            promoteCompletedBuffer()
        }

        advanceTemporalHistories()
        return true
    }

    /// **A `blocking` frame whose CPU wait is a suspension instead of a stall.** Commit the
    /// frame under the full capture contract — never dropped, promoted before the caller
    /// reads `outputTexture` — but return as soon as it is *committed*, leaving the
    /// wait-and-promote to `awaitCommittedFrame()`.
    ///
    /// WHY. `render(blocking: true)` ends in `cb.waitUntilCompleted()`, which parks the
    /// CALLING thread in `__psynch_cvwait` until the GPU is done. For a headless harness that
    /// is exactly right — there is nothing else for that thread to do. For an interactive
    /// host settling a still on the main thread it is not: a 4-second ray-traced frame is
    /// 4 seconds in which the run loop does not iterate, so the window does not repaint, the
    /// cursor does not track and a progress indicator does not move. Rendering one frame per
    /// `await` does not help by itself — the thread is still inside the wait for all but the
    /// microseconds between frames. The wait itself has to move.
    ///
    /// Pair it strictly:
    /// ```swift
    /// _ = renderer.renderAwaited()
    /// await renderer.awaitCommittedFrame()   // outputTexture is now THIS frame
    /// ```
    /// Between the two calls the frame is in flight and `outputTexture` still holds the
    /// PREVIOUS one, so a reader that skips the await gets a frame behind — the same trap
    /// `blocking: false` has always had, and the reason the blocking path exists.
    ///
    /// Not a replacement for `render(blocking:)`: a synchronous caller (every `swift test`
    /// capture, every GPU gate) has no suspension point to spend and should keep using it.
    @discardableResult
    public func renderAwaited() -> Bool {
        gbs.armAwaitedFrame = true
        return render(blocking: true)
    }

    /// Wait for the frame `renderAwaited()` committed, then promote it — the second half of
    /// that call's capture contract, performed from a suspension so the caller's thread stays
    /// in its run loop while the GPU works.
    ///
    /// Safe to call when nothing is pending (no armed render, or one that returned early on an
    /// empty scene): it promotes and returns, which is precisely what the blocking path leaves
    /// behind in those cases. Awaiting the same frame twice is likewise a no-op rather than a
    /// hang — the signal is consumed on the first await.
    public func awaitCommittedFrame() async {
        if let signal = gbs.pendingFrameCompletion {
            gbs.pendingFrameCompletion = nil
            await signal.wait()
        }
        promoteCompletedBuffer()
    }

    func blankSky() {
        guard let cb = commandQueue.makeCommandBuffer() else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = outputTexture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.85, green: 0.86, blue: 0.90, alpha: 1)
        if let enc = timedRenderEncoder(cb, pass, "blankSky") {
            enc.label = "Illuminatorama.blankSky"
            enc.endEncoding()
        }
        cb.commit()
    }
}
