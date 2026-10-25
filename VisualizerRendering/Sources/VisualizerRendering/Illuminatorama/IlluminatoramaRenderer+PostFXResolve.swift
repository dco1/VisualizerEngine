import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Phase 2.7 — TAA resolve. Reprojects last frame's accumulated HDR with
    /// the velocity buffer, neighborhood-clamps it, and blends a small
    /// contribution of the current frame in. Skipped entirely when TAA is
    /// disabled — bloom + tonemap then read directly from hdrCompositeTexture.
    func encodeTAAResolve(_ cb: MTLCommandBuffer) {
        guard taaEnabled else { return }
        guard let enc = timedComputeEncoder(cb, "taa") else { return }
        enc.label = "Illuminatorama.taa"
        enc.setComputePipelineState(pfx.taaResolvePipeline)
        enc.setTexture(pfx.hdrCompositeTexture,     index: 0)
        enc.setTexture(previousHistoryTexture,  index: 1)
        enc.setTexture(pfx.velocityTexture,         index: 2)
        enc.setTexture(currentHistoryTexture,   index: 3)
        // Phase 4.44 — depth lets the resolve synthesise a camera-only motion
        // vector for sky pixels (which the G-buffer leaves at zero velocity).
        enc.setTexture(gbs.depthTexture,            index: 4)
        // Previous frame's depth — lets the resolve reject history at a
        // disocclusion (a fast mover vacated this pixel) even when the vacated
        // surface and the mover share luma, which neighbourhood clamping can't.
        enc.setTexture(gbs.previousDepthTexture,    index: 5)
        // The downstream frame, kept OUT of the accumulator at index 3 so the
        // resolve's acutance sharpen is applied once on the way out instead of
        // compounding inside the history every frame.
        enc.setTexture(pfx.taaResolvedTexture,      index: 6)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        var still = TAAStillParams(still: taaStillAccumulation ? 1 : 0, karis: taaStillKarisWeights ? 1 : 0,
                                   noSharpen: taaStillSharpen ? 0 : 1)
        enc.setBytes(&still, length: MemoryLayout<TAAStillParams>.stride, index: 1)
        dispatch(enc, pipeline: pfx.taaResolvePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// What bloom + tonemap should read from this frame. With TAA on, that's
    /// the history texture we just wrote to; otherwise the SSR-composited HDR.
    private var postFXSource: MTLTexture {
        taaEnabled ? pfx.taaResolvedTexture : pfx.hdrCompositeTexture
    }

    /// The HDR colour post-FX (exposure, DOF, bloom, tonemap) should sample.
    /// After the post-resolve overlay it's the display buffer; otherwise the
    /// plain resolved source.
    var displaySource: MTLTexture {
        pfx.overlayApplied ? gbs.hdrTexture : postFXSource
    }

    /// Phase 4.43 — composite the additive overlay FX (dust motes + emitter
    /// sparks) AFTER the TAA resolve so they never enter the temporal history.
    ///
    /// WHY HERE, NOT EARLIER: these FX are sub-pixel-bright points that move on
    /// their own (curl-noise drift, particle velocity) but write NO motion
    /// vectors into the velocity attachment. Drawn before the resolve (the old
    /// Phase 4.42 placement) the TAA kernel reprojected each mote as if it were
    /// static and blended in ~90% of the previous frame, smearing every mote
    /// into a trail along its path — most visible where the window-shadow test
    /// spikes a mote's brightness in the sunbeam. Compositing past the resolve
    /// keeps the history ping-pong trail-free.
    ///
    /// The display buffer is `hdrTexture`, which is idle once the SSR composite
    /// has consumed it — reused here instead of allocating a fresh full-res
    /// target. We seed it with the resolved opaque image (a blit), then run the
    /// existing additive draws (which already target `hdrTexture` with the
    /// G-buffer depth bound, so opaque geometry still occludes the FX).
    func encodePostResolveFX(_ cb: MTLCommandBuffer) {
        pfx.overlayApplied = false
        let hasEmitters = fxs.particleEmitters.contains { $0.enabled }
        guard !fxs.particleFields.isEmpty || hasEmitters || !fxs.particleStreaks.isEmpty || !fxs.crystalFlakeSources.isEmpty else { return }

        // Seed the display buffer with the resolved opaque image.
        let resolved = postFXSource
        if resolved !== gbs.hdrTexture {
            guard let blit = cb.makeBlitCommandEncoder() else { return }
            blit.label = "Illuminatorama.overlay.seed"
            blit.copy(from: resolved,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: width, height: height, depth: 1),
                      to: gbs.hdrTexture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        }

        // Additive draws into the display buffer (all target `hdrTexture`).
        encodeParticleDraw(cb)
        encodeExternalPointDraw(cb)
        encodeStreakDraw(cb)           // Phase 4.44b — velocity-aligned streak quads
        encodeCrystalFlakeDraw(cb)     // near-camera ice-crystal flakes
        pfx.overlayApplied = true
    }

    /// Source for bloom + tonemap: the DOF output when DOF ran this frame,
    /// otherwise the raw post-FX source. (Exposure estimate still reads the
    /// pre-DOF source so the blur doesn't perturb auto-exposure.)
    /// `hdrPreExposure`, sanitised (non-finite / ≤ 0 ⇒ 1).
    var preExposureDivisor: Float {
        hdrPreExposure.isFinite && hdrPreExposure > 0 ? hdrPreExposure : 1
    }

    var bloomTonemapSource: MTLTexture {
        pfx.localToneMapOutput ?? (dofs.dofApplied ? dofs.dofOutputTexture : nil) ?? displaySource
    }

    // ── Local tone mapping (IlluminatoramaLocalToneMap.swift) ─────────────
    /// Local adaptation of the finished optical image (post-DOF, post-outline) — what bloom,
    /// halation and the tonemap then read. No-op unless `localToneMapping.isEnabled`; the
    /// `VIZ_ILLUMI_HDR_DUMP_PATH` diagnostic dumps the frame it would adapt either way. The
    /// exposure it anchors on is the tonemap's own (`easedExposure` × the metered exposure).
    func encodeLocalToneMap(_ cb: MTLCommandBuffer) {
        let source = bloomTonemapSource
        if IlluminatoramaHDRDump.path != nil {
            hdrDump.encode(cb, source: source, exposureBuffer: pfx.exposureBuffer,
                           hostExposure: pfx.easedExposure / preExposureDivisor, autoExposure: autoExposureEnabled)
        }
        guard localToneMapping.isEnabled else { return }
        pfx.localToneMapOutput = localToneMapPass.encode(
            cb, source: source, exposureBuffer: pfx.exposureBuffer, hostExposure: pfx.easedExposure / preExposureDivisor,
            autoExposure: autoExposureEnabled, settings: localToneMapping,
            makeEncoder: { self.timedComputeEncoder(cb, $0) })
    }

    func encodeTonemapPass(_ cb: MTLCommandBuffer) {
        // Phase 4.28 — RENDER pass (not compute). Draws a fullscreen triangle
        // into `tonemapWriteTarget`; the end-of-pass `.store` is what makes the
        // result visible to SceneKit's cross-queue background sample without a
        // CPU completion wait (the bug that made heavy scenes show a blank /
        // white overlay frame). `.dontCare` load is fine — the fragment writes
        // every output pixel.
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = pfx.tonemapWriteTarget
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let enc = timedRenderEncoder(cb, pass, "tonemap") else { return }
        enc.label = "Illuminatorama.tonemap"
        enc.setRenderPipelineState(pfx.tonemapPipeline)
        enc.setFragmentTexture(bloomTonemapSource, index: 0)
        enc.setFragmentTexture(bloomResultTexture, index: 1)
        // Issue #65 — 3D colour-grade LUT at texture(2). Always bound (identity by
        // default), so the shader never branches on nil; the grade is gated by
        // `colorLUTAmount` in the uniforms.
        enc.setFragmentTexture(pfx.colorLUT, index: 2)
        // Issue #65 — screen-space velocity at texture(3) for motion blur. Always
        // bound; gated by `motionBlurStrength` (0 = the tonemap skips the gather).
        enc.setFragmentTexture(pfx.velocityTexture, index: 3)
        // Issue #65 — G-buffer channels at texture(4..6): albedo / normal /
        // roughness / metalness / depth. Always bound. TWO consumers now read
        // them — the Debug-view readouts (`debugTerm >= 10`) and the diagram
        // look (`diagramParams.x > 0`), which builds its flat fills and its
        // outlines out of exactly these three. Both are gated on their own
        // uniform, so a default photoreal render still samples neither.
        enc.setFragmentTexture(gbs.gbufferAlbedoMet, index: 4)
        enc.setFragmentTexture(gbs.gbufferNormalRgh, index: 5)
        enc.setFragmentTexture(gbs.depthTexture, index: 6)
        // Phase 9 — film LUT at texture(7); 2..6 were taken by the colour-grade
        // LUT / velocity / G-buffer binds in the merge. nil = disabled (the shader
        // checks `frame.filmLUTStrength > 0` before sampling).
        enc.setFragmentTexture(filmLUTTexture, index: 7)
        // Halation halo at texture(8). Always bound (the texture is allocated
        // unconditionally); its contents are stale when halationIntensity == 0,
        // and the shader's branch is gated on the same uniform, so it isn't read then.
        enc.setFragmentTexture(pfx.halationBlurV, index: 8)
        // The AO field at texture(9) for `DebugTerm.ssao` — the same texture the
        // lighting pass multiplies its indirect terms by (post-denoise when the
        // denoiser is on), so the debug view IS the shipped multiplier, not a
        // re-derivation. Always bound; only read when `debugTerm == 16`.
        enc.setFragmentTexture(aoSourceTexture, index: 9)
        enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 0)
        // Phase 4.21 — exposure buffer at buffer(1). The fragment reads
        // `expoState.smoothedExposure` when `frame.autoExposureEnabled`
        // is set, falls back to `frame.exposure` otherwise.
        enc.setFragmentBuffer(pfx.exposureBuffer, offset: 0, index: 1)
        // Fullscreen triangle — 3 vertices, positions synthesised in the
        // vertex shader from `[[vertex_id]]`; no vertex buffer needed.
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    /// Halton low-discrepancy sequence, one dimension. Index 0 is excluded
    /// (the algorithm degenerates there) — callers pass `frameIndex + 1`.
    static func halton(_ index: UInt32, base: UInt32) -> Float {
        var f: Float = 1
        var r: Float = 0
        var i = index
        while i > 0 {
            f /= Float(base)
            r += f * Float(i % base)
            i /= base
        }
        return r
    }
}
