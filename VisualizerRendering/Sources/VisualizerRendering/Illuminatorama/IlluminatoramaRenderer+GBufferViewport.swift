import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
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
            self.gbs.gbufferMaterialLayer = t.materialLayer   // DH-1300
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
}
