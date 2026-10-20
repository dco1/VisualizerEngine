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
    private var displaySource: MTLTexture {
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
        guard !fxs.particleFields.isEmpty || hasEmitters || !fxs.particleStreaks.isEmpty else { return }

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

    /// Lens-accurate depth-of-field on the resolved HDR. Runs after the exposure
    /// estimate and before bloom; writes a private DOF texture that bloom +
    /// tonemap then read. No-op unless `dofEnabled` and the pipelines exist.
    ///
    /// Three dispatches: a max-|CoC| reduction per 16×16 tile, a dilate of that map by the
    /// gather's own reach, then the gather — which reads one dilated texel to learn how far it
    /// must look to find a NEAR-field neighbour whose confusion disc covers it, and to skip
    /// whole tiles that are in focus. See `IlluminatoramaDOF.metal` for the optics.
    func encodeDOFPass(_ cb: MTLCommandBuffer) {
        dofs.dofApplied = false
        if dofReferenceAccumulation {
            encodeDOFReferenceAccumulation(cb)
            return
        }
        guard dofEnabled, dofCoCCoefficient > 0,
              let pipeline = dofs.dofPipeline, let tilePipeline = dofs.dofTilePipeline,
              let dilatePipeline = dofs.dofDilatePipeline else { return }
        // Lazily (re)allocate the DOF target to the current internal size.
        if dofs.dofOutputTexture == nil
            || dofs.dofOutputTexture?.width != width || dofs.dofOutputTexture?.height != height {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = .rgba16Float
            d.width = max(1, width); d.height = max(1, height)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofOutputTexture = device.makeTexture(descriptor: d)
            dofs.dofOutputTexture?.label = "Illuminatorama.dof"
        }
        let tile = Self.dofTileSize
        let tileW = max(1, (width + tile - 1) / tile), tileH = max(1, (height + tile - 1) / tile)
        if dofs.dofTileTexture == nil
            || dofs.dofTileTexture?.width != tileW || dofs.dofTileTexture?.height != tileH {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            // r = max |CoC|, g = the near field's max CoC (the fast gather's bound; unused otherwise).
            d.pixelFormat = .rg32Float
            d.width = tileW; d.height = tileH
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofTileTexture = device.makeTexture(descriptor: d)
            dofs.dofTileTexture?.label = "Illuminatorama.dof.cocTiles"
            dofs.dofTileDilatedTexture = device.makeTexture(descriptor: d)
            dofs.dofTileDilatedTexture?.label = "Illuminatorama.dof.cocTilesDilated"
        }
        // The prefilter pyramid: half resolution, full mip chain. Sized from the internal
        // render size like everything else here, so it follows the draft/settled scale.
        let pw = max(1, (width + 1) / 2), ph = max(1, (height + 1) / 2)
        // The min-CoC pyramid's format: R = least |CoC| (the sharp-subject tests) everywhere; the
        // layered composite (`dofPartialOcclusion`) also reads G/B = the signed-CoC range, so only
        // it pays for an rgba16Float pyramid.
        let minCoCFormat: MTLPixelFormat = dofSharpSubjects && dofPartialOcclusion ? .rgba16Float : .r16Float
        if dofs.dofPrefilterTexture == nil
            || dofs.dofPrefilterTexture?.width != pw || dofs.dofPrefilterTexture?.height != ph
            || dofs.dofMinCoCTexture?.pixelFormat != minCoCFormat {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = .rgba16Float
            d.width = pw; d.height = ph
            d.mipmapLevelCount = Int(log2(Double(max(pw, ph))).rounded(.down)) + 1
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofPrefilterTexture = device.makeTexture(descriptor: d)
            dofs.dofPrefilterTexture?.label = "Illuminatorama.dof.prefilter"
            // The min-CoC pyramid rides along (always allocated with it, so the kernels' slots are
            // always bound; only `dofSharpSubjects` writes or reads it).
            d.pixelFormat = minCoCFormat
            dofs.dofMinCoCTexture = device.makeTexture(descriptor: d)
            dofs.dofMinCoCTexture?.label = "Illuminatorama.dof.minCoC"
            dofs.dofMinCoCLevels = (0..<d.mipmapLevelCount).compactMap { l in
                dofs.dofMinCoCTexture?.makeTextureView(pixelFormat: minCoCFormat, textureType: .type2D,
                                                  levels: l..<(l + 1), slices: 0..<1)
            }
            dofs.dofNearTexture = nil
            dofs.dofNearCoCTexture = nil
        }
        // The layered composite's near-field coverage pyramid (the prefilter's size and mip chain).
        if dofSharpSubjects && dofPartialOcclusion, dofs.dofNearTexture == nil,
           let pre = dofs.dofPrefilterTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = .rgba16Float
            d.width = pre.width; d.height = pre.height
            d.mipmapLevelCount = pre.mipmapLevelCount
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofNearTexture = device.makeTexture(descriptor: d)
            dofs.dofNearTexture?.label = "Illuminatorama.dof.nearField"
            d.pixelFormat = .r16Float
            dofs.dofNearCoCTexture = device.makeTexture(descriptor: d)
            dofs.dofNearCoCTexture?.label = "Illuminatorama.dof.nearCoC"
        }
        guard let out = dofs.dofOutputTexture, let tiles = dofs.dofTileTexture,
              let dilated = dofs.dofTileDilatedTexture else { return }
        // Point taps unless the pyramid is genuinely there to read: the shader treats 0 as
        // "sample the frame directly", which is the correct fallback and NOT a silent one —
        // it is the only value the field ever held before DH-0883.
        let prefilter: MTLTexture? = dofs.dofPrefilterPipeline != nil ? dofs.dofPrefilterTexture : nil
        // The sharp-subject gather needs its min-CoC pyramid (and the prefilter that seeds it).
        let minCoC = dofs.dofMinCoCTexture
        let sharpSubjects = dofSharpSubjects && prefilter != nil && minCoC != nil && dofs.dofMinCoCPipeline != nil
            && dofs.dofMinCoCLevels.count == (minCoC?.mipmapLevelCount ?? -1)
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        var p = DOFParams(
            invProjection: fu.invProjection,
            focusDist: max(0.05, dofFocusDistance),
            cocCoefficient: max(0, dofCoCCoefficient),
            maxRadius: max(1, dofMaxRadius),
            blades: dofBlades >= 3 ? Float(dofBlades) : 0,
            bladeRotation: dofBladeRotation,
            catsEye: max(0, min(1, dofCatsEye)),
            width: UInt32(width), height: UInt32(height),
            tileW: UInt32(tileW), tileH: UInt32(tileH), tileSize: UInt32(tile),
            prefilterScale: prefilter != nil ? max(0, dofPrefilterScale) : 0,
            cocFloor: max(0, dofDiffractionCoCPixels),
            subjectAware: sharpSubjects ? 1 : 0)
        p.partialOcclusion = sharpSubjects && dofPartialOcclusion ? 1 : 0
        let nearOn = p.partialOcclusion > 0.5 && dofs.dofNearTexture != nil && dofs.dofNearCoCTexture != nil
        p.nearPyramid = nearOn ? 1 : 0
        // The near-field slots: the pyramid while it is on, else any texture of the pyramid's size
        // (bound, never read or written — the kernels test `nearPyramid`).
        let nearBinding: MTLTexture? = nearOn ? dofs.dofNearTexture : prefilter
        let nearCoCBinding: MTLTexture? = nearOn ? dofs.dofNearCoCTexture : minCoC
        // The fast gather needs the half-res target (same size as the pyramid's base).
        if sharpSubjects && dofFastGather && dofs.dofHalfPipeline != nil,
           dofs.dofHalfTexture == nil || dofs.dofHalfTexture?.width != pw || dofs.dofHalfTexture?.height != ph {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: pw, height: ph,
                                                             mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofHalfTexture = device.makeTexture(descriptor: d)
            dofs.dofHalfTexture?.label = "Illuminatorama.dof.half"
        }
        let fast = sharpSubjects && dofFastGather && dofs.dofHalfPipeline != nil && dofs.dofHalfTexture != nil
        p.fastGather = fast ? 1 : 0
        p.halfMinCoC = max(1, dofHalfResMinCoC)
        // The quarter tier (opt-in): its target is a quarter of the internal size (the prefilter
        // pyramid's level 1), and it needs the pyramid's mips for the blocks' own colour.
        let qw = max(1, (width + 3) / 4), qh = max(1, (height + 3) / 4)
        if fast && dofQuarterResMinCoC > 0 && dofs.dofQuarterPipeline != nil,
           dofs.dofQuarterTexture == nil || dofs.dofQuarterTexture?.width != qw || dofs.dofQuarterTexture?.height != qh {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: qw, height: qh,
                                                             mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofQuarterTexture = device.makeTexture(descriptor: d)
            dofs.dofQuarterTexture?.label = "Illuminatorama.dof.quarter"
        }
        let quarter = fast && dofQuarterResMinCoC > 0 && dofs.dofQuarterPipeline != nil && dofs.dofQuarterTexture != nil
        p.quarterMinCoC = quarter ? max(6, dofQuarterResMinCoC) : 0
        guard let tileEnc = timedComputeEncoder(cb, "dof.tiles") else { return }
        tileEnc.label = "Illuminatorama.dof.tiles"
        tileEnc.setTexture(gbs.depthTexture, index: 0)
        tileEnc.setTexture(tiles, index: 1)
        tileEnc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        if dofFastGather, let par = dofs.dofTileParallelPipeline {
            // One threadgroup per tile (bit-identical map; the serial kernel walked 256 texels a thread).
            let side = min(tile, max(1, Int(sqrt(Double(par.maxTotalThreadsPerThreadgroup)))))
            tileEnc.setComputePipelineState(par)
            tileEnc.dispatchThreadgroups(MTLSize(width: tileW, height: tileH, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: side, height: side, depth: 1))
        } else {
            tileEnc.setComputePipelineState(tilePipeline)
            dispatch(tileEnc, pipeline: tilePipeline, width: tileW, height: tileH)
        }
        tileEnc.endEncoding()

        // Spread each tile's max by the gather's reach, so the gather itself reads one texel.
        guard let dilEnc = timedComputeEncoder(cb, "dof.dilate") else { return }
        dilEnc.label = "Illuminatorama.dof.dilate"
        dilEnc.setComputePipelineState(dilatePipeline)
        dilEnc.setTexture(tiles, index: 0)
        dilEnc.setTexture(dilated, index: 1)
        dilEnc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        dispatch(dilEnc, pipeline: dilatePipeline, width: tileW, height: tileH)
        dilEnc.endEncoding()

        // Prefilter: the pyramid's base at half resolution, then the blit engine box-filters
        // the rest. Without this each tap point-samples, and a tap's share of a large disc is
        // many pixels wide — so a small bright highlight is either hit or missed per tap and
        // the disc fills with grain. Runs on the SAME source the gather reads.
        if let prePipeline = dofs.dofPrefilterPipeline, let pre = prefilter {
            if let preEnc = timedComputeEncoder(cb, "dof.prefilter") {
                preEnc.label = "Illuminatorama.dof.prefilter"
                preEnc.setComputePipelineState(prePipeline)
                preEnc.setTexture(displaySource, index: 0)
                preEnc.setTexture(gbs.depthTexture, index: 1)
                preEnc.setTexture(pre, index: 2)
                preEnc.setTexture(minCoC, index: 3)
                preEnc.setTexture(nearBinding, index: 4)
                preEnc.setTexture(nearCoCBinding, index: 5)
                preEnc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
                dispatch(preEnc, pipeline: prePipeline, width: pre.width, height: pre.height)
                preEnc.endEncoding()
            }
            if pre.mipmapLevelCount > 1, let blit = cb.makeBlitCommandEncoder() {
                blit.label = "Illuminatorama.dof.prefilter.mips"
                blit.generateMipmaps(for: pre)
                if nearOn, let near = dofs.dofNearTexture, let nearCoC = dofs.dofNearCoCTexture {
                    blit.generateMipmaps(for: near)
                    blit.generateMipmaps(for: nearCoC)
                }
                blit.endEncoding()
            }
            // Min-CoC levels 1… from the prefilter-written base (sharp-subject mode only).
            if sharpSubjects, let minP = dofs.dofMinCoCPipeline, dofs.dofMinCoCLevels.count > 1,
               let mEnc = timedComputeEncoder(cb, "dof.minCoC") {
                mEnc.label = "Illuminatorama.dof.minCoC"
                mEnc.setComputePipelineState(minP)
                for l in 1..<dofs.dofMinCoCLevels.count {
                    let dst = dofs.dofMinCoCLevels[l]
                    mEnc.setTexture(dofs.dofMinCoCLevels[l - 1], index: 0)
                    mEnc.setTexture(dst, index: 1)
                    dispatch(mEnc, pipeline: minP, width: dst.width, height: dst.height)
                }
                mEnc.endEncoding()
            }
        }

        // The quarter tier first (opt-in): the half-res pass reads it to skip the texels it serves.
        if quarter, let quarterP = dofs.dofQuarterPipeline, let qTex = dofs.dofQuarterTexture,
           let qEnc = timedComputeEncoder(cb, "dof.quarter") {
            qEnc.label = "Illuminatorama.dof.quarter"
            qEnc.setComputePipelineState(quarterP)
            qEnc.setTexture(displaySource, index: 0)
            qEnc.setTexture(gbs.depthTexture, index: 1)
            qEnc.setTexture(qTex, index: 2)
            qEnc.setTexture(dilated, index: 3)
            qEnc.setTexture(prefilter, index: 4)
            qEnc.setTexture(minCoC, index: 5)
            qEnc.setTexture(nearBinding, index: 6)
            qEnc.setTexture(nearCoCBinding, index: 7)
            qEnc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
            dispatch(qEnc, pipeline: quarterP, width: qTex.width, height: qTex.height)
            qEnc.endEncoding()
        }
        // The quarter result, or any half-precision texture when the tier is off (bound, unread).
        let quarterBinding: MTLTexture? = quarter ? dofs.dofQuarterTexture : prefilter

        if fast, let halfP = dofs.dofHalfPipeline, let half = dofs.dofHalfTexture,
           let hEnc = timedComputeEncoder(cb, "dof.half") {
            hEnc.label = "Illuminatorama.dof.half"
            hEnc.setComputePipelineState(halfP)
            hEnc.setTexture(displaySource, index: 0)
            hEnc.setTexture(gbs.depthTexture, index: 1)
            hEnc.setTexture(half, index: 2)
            hEnc.setTexture(dilated, index: 3)
            hEnc.setTexture(prefilter, index: 4)
            hEnc.setTexture(minCoC, index: 5)
            hEnc.setTexture(quarterBinding, index: 6)
            hEnc.setTexture(nearBinding, index: 7)
            hEnc.setTexture(nearCoCBinding, index: 8)
            hEnc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
            dispatch(hEnc, pipeline: halfP, width: half.width, height: half.height)
            hEnc.endEncoding()
        }

        guard let enc = timedComputeEncoder(cb, "dof") else { return }
        enc.label = "Illuminatorama.dof"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(displaySource, index: 0)
        enc.setTexture(gbs.depthTexture, index: 1)
        enc.setTexture(out, index: 2)
        enc.setTexture(dilated, index: 3)
        enc.setTexture(prefilter, index: 4)
        enc.setTexture(minCoC, index: 5)
        // The fast gather's half-res result (any half-size texture when it is off: bound, unread).
        enc.setTexture(fast ? dofs.dofHalfTexture : prefilter, index: 6)
        enc.setTexture(quarterBinding, index: 7)
        enc.setTexture(nearBinding, index: 8)
        enc.setTexture(nearCoCBinding, index: 9)
        enc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
        dofs.dofApplied = true
    }

    /// Swift mirror of the Metal `DOFReferenceParams` (32 bytes).
    private struct DOFReferenceParams {
        var aperture: SIMD2<Float>
        var catsEye: Float
        var add: Float
        var reset: Float
        var width: UInt32
        var height: UInt32
        var _pad: UInt32 = 0
    }

    /// `dofReferenceAccumulation`: add this frame's pinhole HDR into the float accumulator (or
    /// re-present it) and hand the running mean to the post chain in the DOF output's place.
    private func encodeDOFReferenceAccumulation(_ cb: MTLCommandBuffer) {
        guard let pipeline = dofReferencePipeline else { return }
        if dofs.dofOutputTexture == nil
            || dofs.dofOutputTexture?.width != width || dofs.dofOutputTexture?.height != height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: max(1, width),
                                                             height: max(1, height), mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofOutputTexture = device.makeTexture(descriptor: d)
            dofs.dofOutputTexture?.label = "Illuminatorama.dof"
        }
        if dofs.dofReferenceAccumTexture == nil
            || dofs.dofReferenceAccumTexture?.width != width || dofs.dofReferenceAccumTexture?.height != height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: max(1, width),
                                                             height: max(1, height), mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofs.dofReferenceAccumTexture = device.makeTexture(descriptor: d)
            dofs.dofReferenceAccumTexture?.label = "Illuminatorama.dof.reference"
            dofs.dofReferenceNeedsReset = true
        }
        guard let out = dofs.dofOutputTexture, let accum = dofs.dofReferenceAccumTexture,
              let enc = timedComputeEncoder(cb, "dof.reference") else { return }
        let reset = dofs.dofReferenceNeedsReset
        let add = dofReferenceAddsFrame
        var p = DOFReferenceParams(aperture: dofReferenceApertureSample, catsEye: max(0, min(1, dofCatsEye)),
                                   add: add ? 1 : 0, reset: reset ? 1 : 0,
                                   width: UInt32(width), height: UInt32(height))
        enc.label = "Illuminatorama.dof.reference"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(displaySource, index: 0)
        enc.setTexture(accum, index: 1)
        enc.setTexture(out, index: 2)
        enc.setBytes(&p, length: MemoryLayout<DOFReferenceParams>.stride, index: 0)
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
        if reset { dofReferenceFrameCount = 0; dofs.dofReferenceNeedsReset = false }
        if add { dofReferenceFrameCount += 1 }
        dofs.dofApplied = true
    }

    func encodeExposureEstimate(_ cb: MTLCommandBuffer) {
        guard autoExposureEnabled else { return }
        // Held (opt-in): the smoothed exposure in `exposureBuffer` stays exactly where it is.
        guard !autoExposureHold else { return }
        // Per-frame dt, clamped to a sane window so a stalled frame
        // (debugger pause, scene reload) doesn't pump the EMA.
        let now = CACurrentMediaTime()
        gbs.lastFrameDuration = max(0.001, min(0.5, fixedFrameDelta ?? (now - pfx.lastExposureTickTime)))
        pfx.lastExposureTickTime = now
        guard let enc = timedComputeEncoder(cb, "exposureEstimate") else { return }
        enc.label = "Illuminatorama.exposureEstimate"
        enc.setComputePipelineState(pfx.exposureEstimatePipeline)
        enc.setTexture(displaySource, index: 0)
        enc.setBuffer(pfx.exposureBuffer, offset: 0, index: 0)
        // imgSize for stride computation inside the kernel (the meter's coarse grid; opt-in
        // `exposureMeterGridSize` pins it independently of the internal render scale).
        let grid = exposureMeterGridSize.map { SIMD2(max(8, $0.x), max(8, $0.y)) } ?? SIMD2(width, height)
        var imgSize = SIMD2<UInt32>(UInt32(grid.x), UInt32(grid.y))
        enc.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        // params: x = targetEV, y = halfLife, z = maxBoost, w = minBoost.
        // Phase 4.28 — minBoost is now an independent floor (not `1/maxBoost`)
        // so bright `environmentImage` scenes can darken far enough to escape
        // a blown-white frame while dim scenes keep their ≤maxBoost cap.
        var params = SIMD4<Float>(autoExposureTargetEV,
                                   autoExposureHalfLife,
                                   autoExposureMaxBoost,
                                   max(autoExposureMinExposure, 0.01))
        enc.setBytes(&params, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
        // params2: x = highlight protection (0 = OFF, an exact no-op inside the
        // kernel), y = the EV the frame's bright half is asked to land at. See
        // `autoExposureHighlightProtection` for why the statistic is the upper-half
        // mean rather than a percentile.
        // w = the HDR pre-exposure the samples are divided by (0 ⇒ none — `hdrPreExposure`).
        var params2 = SIMD4<Float>(max(0, min(1, autoExposureHighlightProtection)),
                                   autoExposureHighlightEV,
                                   exposureHistogramEnabled ? 1 : 0,
                                   preExposureDivisor == 1 ? 0 : preExposureDivisor)
        enc.setBytes(&params2, length: MemoryLayout<SIMD4<Float>>.stride, index: 3)
        // params3 (DH-0655): x = metering (0 mean, 1 percentile), y = key percentile,
        // z = guard percentile (0 = none), w = guard EV. Mean + instrument off ⇒ no histogram.
        var params3 = SIMD4<Float>(autoExposureMetering == .percentile ? 1 : 0,
                                   max(0, min(1, autoExposureKeyPercentile)),
                                   max(0, min(1, autoExposureGuardPercentile)),
                                   autoExposureGuardEV)
        enc.setBytes(&params3, length: MemoryLayout<SIMD4<Float>>.stride, index: 4)
        enc.setBuffer(pfx.exposureHistogramBuffer, offset: 0, index: 5)
        // Update the host-driven `dt` slot in the buffer — the EMA step
        // inside the kernel needs to know how much wall-time elapsed
        // since the last estimate so the half-life math is correct.
        let dt = Float(max(0.001, gbs.lastFrameDuration))
        // The slot is 4 floats in; ExposureState.deltaTime is offset 12.
        let dtPtr = pfx.exposureBuffer.contents().advanced(by: 12)
                                              .bindMemory(to: Float.self, capacity: 1)
        dtPtr.pointee = dt
        // Single threadgroup of 256 threads with shared memory for sum +
        // count. `threadgroupMemoryLength(at:)` is bytes for one Float
        // per thread; two arrays = 2 × 256 × 4 = 2048 bytes for floats
        // and 1024 for uints. Set both indices.
        enc.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.stride, index: 0)
        enc.setThreadgroupMemoryLength(256 * MemoryLayout<UInt32>.stride, index: 1)
        enc.setThreadgroupMemoryLength(ExposureHistogram.bins * MemoryLayout<UInt32>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let groups = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// The texture the tonemap samples as "bloom": the TOP of the pyramid, at
    /// internal/2. `bloomUpChain` is empty only in the degenerate one-level case
    /// (a frame too small for a second mip), where the prefilter output is the
    /// whole effect.
    private var bloomResultTexture: MTLTexture {
        pfx.bloomUpChain.first ?? pfx.bloomDownChain[0]
    }

    /// The Karis / Call-of-Duty dual-filter pyramid (S1.2): prefilter the HDR to
    /// mip0, box-downsample to the top of the chain, then tent-upsample back
    /// down, blending CONVEXLY with each down level on the way (see the shader's
    /// section header for why convex and not additive).
    ///
    /// One encoder for the whole chain. `makeComputeCommandEncoder()` is a SERIAL
    /// dispatch encoder, so Metal inserts the level-to-level barriers itself —
    /// each dispatch reads a resource the previous one wrote.
    func encodeBloomPasses(_ cb: MTLCommandBuffer) {
        guard let enc = timedComputeEncoder(cb, "bloom") else { return }
        enc.label = "Illuminatorama.bloom"
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)

        // Prefilter — full-res HDR → mip0. The ONLY level that thresholds, and
        // the only one that applies the Karis firefly weighting.
        let mip0 = pfx.bloomDownChain[0]
        enc.setComputePipelineState(pfx.bloomPrefilterPipeline)
        enc.setTexture(bloomTonemapSource, index: 0)
        enc.setTexture(mip0, index: 1)
        dispatch(enc, pipeline: pfx.bloomPrefilterPipeline, width: mip0.width, height: mip0.height)

        // Down chain — each level is a 13-tap box of the one above it.
        enc.setComputePipelineState(pfx.bloomDownPipeline)
        for level in 1..<pfx.bloomDownChain.count {
            let dst = pfx.bloomDownChain[level]
            enc.setTexture(pfx.bloomDownChain[level - 1], index: 0)
            enc.setTexture(dst, index: 1)
            dispatch(enc, pipeline: pfx.bloomDownPipeline, width: dst.width, height: dst.height)
        }

        // DH-0998 — the physical glare's far field: an exact angular convolution over one coarse
        // down level (see `illumi_bloom_glare_tail`), folded into the finest up step below.
        var tail: MTLTexture?
        var tailScale: Float = 0
        if let fit = lastBloomGlareFit, fit.energy > 0, !pfx.bloomUpChain.isEmpty,
           case let src = pfx.bloomDownChain[min(fit.tailGridLevel, pfx.bloomDownChain.count - 1)],
           let out = glareTailOutput(width: max(1, src.width / 2), height: max(1, src.height / 2)) {
            encodeGlareTail(enc, fit: fit, tanHalf: pfx.glareTailTanHalf, source: src, output: out)
            enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
            tail = out
            tailScale = Float(1 / fit.energy)
        }

        // Up chain — coarsest to finest. The first step is seeded from the
        // smallest DOWN level (which is why the up chain is one texture shorter).
        enc.setComputePipelineState(pfx.bloomUpPipeline)
        let last = pfx.bloomDownChain.count - 1
        let levelScatters = lastBloomGlareFit.map { fit in
            (0..<pfx.bloomDownChain.count).map { $0 < fit.scatters.count ? fit.scatters[$0] : 1 }
        } ?? Self.bloomPSFScatters(levels: pfx.bloomDownChain.count, exponent: bloomPSFExponent)
        for level in stride(from: last - 1, through: 0, by: -1) {
            let low = (level == last - 1) ? pfx.bloomDownChain[last] : pfx.bloomUpChain[level + 1]
            let dst = pfx.bloomUpChain[level]
            enc.setTexture(low, index: 0)
            enc.setTexture(pfx.bloomDownChain[level], index: 1)
            enc.setTexture(dst, index: 2)
            var s: Float = levelScatters?[level] ?? -1
            enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 1)
            var ts: Float = level == 0 ? tailScale : 0
            enc.setTexture(level == 0 ? (tail ?? low) : low, index: 3)
            enc.setBytes(&ts, length: MemoryLayout<Float>.stride, index: 2)
            dispatch(enc, pipeline: pfx.bloomUpPipeline, width: dst.width, height: dst.height)
        }
        enc.endEncoding()
    }

    /// The tail's output texture at `width × height` (rgba32Float — it is tiny), reused per size.
    /// Half the source grid in each axis (see `IlluminatoramaGlareSpread.Fit.tailGridLevel`).
    private func glareTailOutput(width: Int, height: Int) -> MTLTexture? {
        if let t = pfx.glareTailTexture, t.width == width, t.height == height { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: width, height: height,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        pfx.glareTailTexture = device.makeTexture(descriptor: d)
        pfx.glareTailTexture?.label = "Illuminatorama.bloom.glareTail"
        return pfx.glareTailTexture
    }

    /// Mirrors `BloomGlareTailParams` in IlluminatoramaPost.metal.
    private struct BloomGlareTailParams {
        var tanHalf: SIMD2<Float>
        var rhoA: Float, rhoB: Float, thetaMax: Float
        var ageTerm: Float = 0, pigment: Float = 0
        var model: Float = 0, lensK: Float = 0, lensInvL2: Float = 0, lensHalfS: Float = 0, pad: Float = 0
    }

    /// Encode `illumi_bloom_glare_tail` over `source` into `output`. Leaves buffer 0 bound to the
    /// tail's parameters — the caller rebinds the frame uniforms.
    private func encodeGlareTail(_ enc: MTLComputeCommandEncoder, fit: IlluminatoramaGlareSpread.Fit,
                                 tanHalf: SIMD2<Float>, source: MTLTexture, output: MTLTexture) {
        guard let pipe = bloomGlareTailPipeline else { return }
        // The ramp is fitted in the pyramid's units — mip0 texels × `texelDegrees`, where
        // `texelDegrees` is the tan-plane step of one mip0 texel expressed in degrees (2/P11 ÷ mip0
        // height) — and applied in image-plane distance, so its degrees convert back by π/180
        // exactly (see the shader for why not in angle).
        let tanPerDegree = Double.pi / 180
        var p = BloomGlareTailParams(tanHalf: tanHalf, rhoA: Float(fit.tailStartDegrees * tanPerDegree),
                                     rhoB: Float(fit.tailEndDegrees * tanPerDegree), thetaMax: Float(fit.tailMaxDegrees))
        // The tail evaluates the SAME model the pyramid's weights were solved against — the fit's.
        switch fit.model {
        case let .eye(age, pigment):
            p.ageTerm = Float(1 + pow(age / 62.5, 4)); p.pigment = Float(pigment)
        case let .lens(L):
            p.model = 1; p.lensK = Float(Double(L.surfaces) * L.b0)
            p.lensInvL2 = Float(1 / (L.l * L.l)); p.lensHalfS = Float(L.s / 2)
        }
        enc.setComputePipelineState(pipe)
        enc.setTexture(source, index: 0)
        enc.setTexture(output, index: 1)
        enc.setBytes(&p, length: MemoryLayout<BloomGlareTailParams>.stride, index: 0)
        // One threadgroup of `kBloomGlareTailThreads` (256) per output cell — see the shader.
        enc.dispatchThreadgroups(MTLSize(width: output.width, height: output.height, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// **TEST INSTRUMENT (Daydream DH-1001): the bloom chain's impulse response, on the GPU.**
    /// Runs the SAME prefilter / down / up pipelines the frame does, in physical-glare mode (no
    /// threshold, no Karis), on a `2·size × 2·size` HDR texture holding one texel of `impulse` at
    /// full-res pixel `at`, with the up-chain's convex weights set to `levelWeights` (Σ = 1; a
    /// one-hot vector isolates one level's kernel Kᵢ). Returns the pyramid output (mip0,
    /// `size × size`) as linear luma, unscaled. Synchronous; call between frames. Uses the current
    /// frame uniforms for everything the bloom kernels do not read.
    public func measureBloomImpulseResponse(levelWeights: [Float], size: Int,
                                            at: SIMD2<Int>, impulse: Float = 1024) -> [Float]? {
        measureBloomImpulseResponse(levelWeights: levelWeights, width: size, height: size, at: at,
                                    impulse: impulse)?.luma
    }

    /// The same instrument over a `width × height` mip0 (a REAL frame's geometry — DH-0998), and
    /// optionally with the far-field glare tail of `glareTail.fit` folded into the finest up step
    /// exactly as the frame does (`tanHalf` = the frame's half-FOV tangents). Returns the pyramid
    /// output's luma (÷ impulse, unscaled — × `fit.energy` gives the glare) and its alpha (the
    /// tail's in-frame share M of each pixel's own light; 1 when no tail runs).
    public func measureBloomImpulseResponse(levelWeights: [Float], width: Int, height: Int,
                                            at: SIMD2<Int>, impulse: Float = 1024,
                                            glareTail: (fit: IlluminatoramaGlareSpread.Fit, tanHalf: SIMD2<Float>)? = nil)
        -> (luma: [Float], alpha: [Float])? {
        let levels = min(levelWeights.count, Self.bloomLevelCount(halfW: width, halfH: height))
        guard levels >= 1, min(width, height) >= 8 else { return nil }
        func tex(_ w: Int, _ h: Int, _ fmt: MTLPixelFormat = .rgba16Float) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        // Input: zero, one impulse texel. Uploaded through a shared staging buffer.
        let fw = 2 * width, fh = 2 * height
        guard let input = tex(fw, fh),
              let stage = device.makeBuffer(length: fw * fh * 8, options: .storageModeShared) else { return nil }
        memset(stage.contents(), 0, fw * fh * 8)
        let hv = Self.float32to16(impulse)
        let px = stage.contents().advanced(by: (at.y * fw + at.x) * 8).assumingMemoryBound(to: UInt16.self)
        px[0] = hv; px[1] = hv; px[2] = hv; px[3] = Self.float32to16(1)
        var down: [MTLTexture] = [], up: [MTLTexture] = []
        for level in 0..<levels {
            let (lw, lh) = Self.bloomLevelSize(halfW: width, halfH: height, level: level)
            guard let d = tex(lw, lh) else { return nil }
            down.append(d)
            if level < levels - 1 {
                guard let u = tex(lw, lh, level == 0 ? .rgba32Float : .rgba16Float) else { return nil }
                up.append(u)
            }
        }
        // A one-level pyramid has no up chain: mip0 itself (half) is read back.
        let readTex: MTLTexture = levels == 1 ? down[0] : up[0]
        let count = width * height
        guard let readBuf = device.makeBuffer(length: count * 16, options: .storageModeShared),
              let ubuf = device.makeBuffer(length: MemoryLayout<IlluminatoramaFrameUniforms>.stride,
                                           options: .storageModeShared),
              let cb = commandQueue.makeCommandBuffer() else { return nil }
        var u = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        u.bloomThreshold = 0
        u.bloomParams = SIMD4(0, 0, 1, 1)
        ubuf.contents().storeBytes(of: u, as: IlluminatoramaFrameUniforms.self)
        let scat = IlluminatoramaGlareSpread.convexScatters(levelWeights.prefix(levels).map(Double.init))

        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: stage, sourceOffset: 0, sourceBytesPerRow: fw * 8, sourceBytesPerImage: fw * fh * 8,
                      sourceSize: MTLSize(width: fw, height: fh, depth: 1),
                      to: input, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin())
            blit.endEncoding()
        }
        guard let enc = cb.makeComputeCommandEncoder() else { return nil }
        enc.setBuffer(ubuf, offset: 0, index: 0)
        enc.setComputePipelineState(pfx.bloomPrefilterPipeline)
        enc.setTexture(input, index: 0); enc.setTexture(down[0], index: 1)
        dispatch(enc, pipeline: pfx.bloomPrefilterPipeline, width: down[0].width, height: down[0].height)
        enc.setComputePipelineState(pfx.bloomDownPipeline)
        for level in stride(from: 1, to: levels, by: 1) {
            enc.setTexture(down[level - 1], index: 0); enc.setTexture(down[level], index: 1)
            dispatch(enc, pipeline: pfx.bloomDownPipeline, width: down[level].width, height: down[level].height)
        }
        var tailTex: MTLTexture?
        var tailScale: Float = 0
        if levels > 1, let g = glareTail, g.fit.energy > 0 {
            let src = down[min(g.fit.tailGridLevel, levels - 1)]
            if let out = tex(max(1, src.width / 2), max(1, src.height / 2), .rgba32Float) {
                encodeGlareTail(enc, fit: g.fit, tanHalf: g.tanHalf, source: src, output: out)
                enc.setBuffer(ubuf, offset: 0, index: 0)
                tailTex = out
                tailScale = Float(1 / g.fit.energy)
            }
        }
        if levels > 1 {
            enc.setComputePipelineState(pfx.bloomUpPipeline)
            let last = levels - 1
            for level in stride(from: last - 1, through: 0, by: -1) {
                let low = level == last - 1 ? down[last] : up[level + 1]
                enc.setTexture(low, index: 0)
                enc.setTexture(down[level], index: 1)
                enc.setTexture(up[level], index: 2)
                var s = scat[level]
                enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 1)
                var ts: Float = level == 0 ? tailScale : 0
                enc.setTexture(level == 0 ? (tailTex ?? low) : low, index: 3)
                enc.setBytes(&ts, length: MemoryLayout<Float>.stride, index: 2)
                dispatch(enc, pipeline: pfx.bloomUpPipeline, width: up[level].width, height: up[level].height)
            }
        }
        enc.endEncoding()
        let src: MTLTexture = readTex
        let bpp = src.pixelFormat == .rgba32Float ? 16 : 8
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: src, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                      sourceSize: MTLSize(width: width, height: height, depth: 1),
                      to: readBuf, destinationOffset: 0, destinationBytesPerRow: width * bpp,
                      destinationBytesPerImage: count * bpp)
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        lastBloomImpulseGPUMilliseconds = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        var out = [Float](repeating: 0, count: count), alpha = out
        if bpp == 16 {
            let f = readBuf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count {
                out[i] = (0.2126 * f[4 * i] + 0.7152 * f[4 * i + 1] + 0.0722 * f[4 * i + 2]) / impulse
                alpha[i] = f[4 * i + 3]
            }
        } else {
            let hp = readBuf.contents().assumingMemoryBound(to: UInt16.self)
            func h2f(_ h: UInt16) -> Float {
                let s: UInt32 = UInt32(h & 0x8000) << 16, e = Int((h >> 10) & 0x1F), m = UInt32(h & 0x3FF)
                if e == 0 { return (s != 0 ? -1 : 1) * Float(m) * pow(2, -24) }
                if e == 31 { return Float(bitPattern: s | 0x7F80_0000 | (m << 13)) }
                return Float(bitPattern: s | UInt32(e - 15 + 127) << 23 | (m << 13))
            }
            for i in 0..<count {
                out[i] = (0.2126 * h2f(hp[4 * i]) + 0.7152 * h2f(hp[4 * i + 1]) + 0.0722 * h2f(hp[4 * i + 2])) / impulse
                alpha[i] = h2f(hp[4 * i + 3])
            }
        }
        return (out, alpha)
    }

    /// Film halation: half-res threshold → dense four-scale separable gaussian, tinted
    /// and added back by the tonemap fragment. Encoded ONLY when the eased intensity is
    /// non-zero — and that's the SAME value the shader's branch reads, so "passes ran"
    /// and "tonemap consumes the halo" can never disagree. A scene that never opts in
    /// pays no dispatches and renders byte-identically.
    func encodeHalationPasses(_ cb: MTLCommandBuffer) {
        guard pfx.easedHalationIntensity > 0 else { return }
        let halfW = pfx.halationBright.width
        let halfH = pfx.halationBright.height
        guard let enc = timedComputeEncoder(cb, "halation") else { return }
        enc.label = "Illuminatorama.halation"

        enc.setComputePipelineState(pfx.halationThresholdPipeline)
        enc.setTexture(bloomTonemapSource, index: 0)
        enc.setTexture(pfx.halationBright, index: 1)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: pfx.halationThresholdPipeline, width: halfW, height: halfH)

        enc.setComputePipelineState(pfx.halationBlurHPipeline)
        enc.setTexture(pfx.halationBright, index: 0)
        enc.setTexture(pfx.halationBlurH, index: 1)
        dispatch(enc, pipeline: pfx.halationBlurHPipeline, width: halfW, height: halfH)

        enc.setComputePipelineState(pfx.halationBlurVPipeline)
        enc.setTexture(pfx.halationBlurH, index: 0)
        enc.setTexture(pfx.halationBlurV, index: 1)
        dispatch(enc, pipeline: pfx.halationBlurVPipeline, width: halfW, height: halfH)
        enc.endEncoding()
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
}

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

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// Per-level scatter for the up-chain that realises `bloomPSFExponent` (nil ⇒ the shared
    /// scatter). Level i's down texture ends up weighted (1−s_i)·Π_{k<i} s_k, so for target
    /// weights a_i (Σ = 1): s_i = 1 − a_i / (1 − Σ_{k<i} a_k). Public for the gate.
    public static func bloomPSFScatters(levels: Int, exponent: Float) -> [Float]? {
        guard exponent > 0, levels >= 2 else { return nil }
        let raw = (0..<levels).map { i in pow(Float(1 << (i + 1)), 2 - exponent) }
        let total = raw.reduce(0, +)
        let a = raw.map { $0 / total }
        var remaining: Float = 1
        var s: [Float] = []
        for i in 0..<levels {
            s.append(remaining > 1e-6 ? min(1, max(0, 1 - a[i] / remaining)) : 1)
            remaining -= a[i]
        }
        return s
    }

    /// Mirror of `TAAStillParams` (IlluminatoramaTAA.metal), 16 bytes.
    struct TAAStillParams {
        var still: UInt32; var karis: UInt32; var noSharpen: UInt32 = 0; var reserved1: UInt32 = 0
    }

    /// **Which scene → display rendering the tonemap applies** (DH-0881).
    ///
    /// The shipped one, `.acesRec709`, is Narkowicz's fitted RRT+ODT curve applied PER CHANNEL
    /// in Rec.709 primaries — no AP0/AP1 transform anywhere, i.e. ACES in name only. A
    /// per-channel curve desaturates by driving the dominant channel into its shoulder first,
    /// so hue rotates as intensity rises: saturated blue → purple, hot red → orange. The other
    /// two options fix that; `.acesRec709` stays the default and is byte-identical.
    public enum DisplayTransform: UInt32, Sendable, CaseIterable, Equatable, Hashable {
        /// The shipped curve, per channel in Rec.709. Byte-identical to every prior build.
        case acesRec709 = 0
        /// The SAME curve evaluated in ACEScg (AP1) and brought back. Same contrast, same
        /// shoulder; the channel clipping now happens in a wide gamut, so hue holds. The
        /// minimal, conservative fix.
        case acesAP1 = 1
        /// AgX — an inset matrix that desaturates before the sigmoid (so no channel enters the
        /// shoulder alone, the blue-light case) and a log2-domain contrast curve. What Blender
        /// moved to, for these reasons.
        case agx = 2
    }

    // ── DH-0655 — the luminance histogram and percentile metering ────────────
    //
    // The mean meter cannot see what it is about to clip; a percentile can, but needs a
    // distribution the reduction never had. `illumi_exposure_estimate` now builds a 64-bin
    // log-luminance histogram (0.25 EV a bin over its [−8, +8] clamp) from the SAME samples the
    // mean averages, when asked. Both levers default OFF and the shipped meter is untouched.

    /// How the auto-exposure target is chosen. `.mean` (default) is the geometric mean it has
    /// always been. `.percentile` meters `autoExposureKeyPercentile` of the log-luminance
    /// distribution instead, and — when `autoExposureGuardPercentile > 0` — caps exposure so
    /// that percentile lands at `autoExposureGuardEV` (a `min`, never a boost).
    public enum AutoExposureMetering: Sendable { case mean, percentile }

    /// One frame's histogram as the meter saw it. Log-luminance is log2 of the metered
    /// brightness (max of luma and half the max channel), pre-exposure.
    public struct ExposureHistogram: Sendable {
        public static let bins = 64, minLogLum: Float = -8, binEV: Float = 0.25
        public let counts: [Float]
        public let total: Float
        public let p50: Float, p95: Float, p99: Float
        /// The kernel's own geometric-mean statistic, for comparing the two meters.
        public let meanLogLum: Float
        /// The percentile the key asked for.
        public let keyLogLum: Float
    }

    // ── Per-term split-render diagnostic ─────────────────────────────
    /// Isolates ONE lighting term in the deferred kernel so a flooded /
    /// flat scene can be decomposed (which term dominates the colour).
    /// `.normal` is the production composite. NOTE: the SSR and any planar-RT
    /// reflections composite in *separate* passes after this kernel, so to
    /// isolate them, A/B `ssrIntensity` rather than this enum.
    public enum DebugTerm: UInt32 {
        case normal = 0, directSun = 1, pointLights = 2, spotLights = 3
        case diffuseIBL = 4, specularIBL = 5, emission = 6, ambient = 7
        /// Isolates the RT surface-cache contribution (GI + reflection atlas
        /// reads) in the instanced/TLAS path — REPLACES the composite with just
        /// that term so a moving object's stale-vs-fresh cache is pixel-obvious.
        /// Unlike cases 1–7 (deferred-shading terms in `illumi_lighting`) this is
        /// wired in `illumi_rt_lighting_tlas`; it only shows with RT + surface
        /// cache on. See docs/illuminatorama/surface-cache-incremental-invalidation.md.
        case surfaceCacheGI = 8
        /// Isolates the surface cache's per-texel VARIANCE (E[L²] − μ², tracked in
        /// the atlas .w channel) at GI/reflection hits — a convergence heatmap:
        /// freshly-reset/cold cards bright, long-static cards near-black. Phase 5 /
        /// B0 instrumentation for the cache-domain denoiser. TLAS path only, like
        /// `surfaceCacheGI`. See docs/illuminatorama/phase5-radiance-cache-streaming.md.
        case surfaceCacheVariance = 9
        /// Raw G-buffer channels (issue #65). Unlike 1–9 these are NOT lighting
        /// terms: the deferred kernel's `switch` defaults to the full composite
        /// for them, and a debug branch in `illumi_tonemap_fs` instead REPLACES
        /// the image with the stored G-buffer channel *before* tonemapping, so
        /// the readout is the true value (albedo/normal/roughness/metalness/
        /// depth/velocity). Drives the app's Debug-view picker; exact no-op for
        /// `.normal` (0). `velocity` reuses the same colourisation as the env
        /// `VIZ_ILLUMI_MV_DEBUG` motion-vector overlay.
        case albedo = 10, gbufferNormal = 11, roughness = 12
        case metalness = 13, depth = 14, velocity = 15
        /// RAW ambient-occlusion field — the exact half-res value `illumi_lighting`
        /// multiplies the indirect terms by, read BEFORE tonemapping so the readout
        /// is the true multiplier and not a tone-curve-compressed ratio. Greyscale:
        /// white = 1.0 (unoccluded), black = 0 (fully occluded).
        ///
        /// This is the instrument that indicted `kSsaoKernel` (2026-08-05): a flat
        /// wall in the middle of a room MUST read ≈1.0 here, and before the
        /// hemisphere fix it read ≈0.44 — the whole room sat on an occlusion
        /// pedestal. Composite views cannot show that, because albedo, the
        /// illumination gradient and the tonemapper all ride on top of it. Reach for
        /// this before theorising about any AO complaint.
        case ssao = 16
        /// Isolates the per-material **cloth sheen** lobe (velvet / linen / wool — strength
        /// arrives packed as a NEGATIVE `emission.a`, see `IlluminatoramaGBuffer.metal`).
        /// Unlike 10–16 this is a LIGHTING term, not a G-buffer channel: `illumi_lighting`
        /// handles it and the tonemap's `>= 10` G-buffer branch deliberately has no case for
        /// it, so it falls through and the term is tonemapped like cases 1–7.
        ///
        /// This is the instrument that measured gap #8 (2026-08-09): before the lobe moved
        /// into `brdf()` it was evaluated ONCE, after the point/spot loops had closed, from
        /// the sun and `frame.ambientColor` alone — so a lamp-lit sofa received exactly zero
        /// sheen and read as plaster. A night-with-lamps frame MUST show fabric here.
        case clothSheen = 17
        /// **Chart-assignment overlay** (DH-0653): REPLACES the composite with one hashed
        /// colour per surface-cache CARD at each pixel's own (primary) surface — the chart
        /// decomposition `VIZ_SURFCACHE_CHARTS` builds, made visible, so merge quality and
        /// residual GI seams are read off topology instead of eyeballed off the composite.
        /// A non-resident card keeps its hue at near-black; near-black grey = no card.
        /// Wired in `illumi_rt_lighting_tlas` like 8/9, so it needs RT + the surface cache
        /// (a TLAS scene). The per-frame hit/miss numbers that go with it are in the
        /// `VIZ_ILLUMI_SURFCACHE_STATS_PATH` sidecar (`hits:` / `atlas:` lines).
        case surfaceCacheCharts = 18
    }

    /// Set the film stock's colour cube.
    ///
    /// The size is read OFF THE TEXTURE, not passed in and not assumed: the cube used to
    /// be a hardcoded 16 on both sides, so shipping the 33-cube the stocks are actually
    /// authored at would have silently sampled a 33³ texture on 16-cell coordinates —
    /// wrong everywhere, and wrong in a way that reads as a grade rather than as a bug.
    /// - Parameters:
    ///   - texture: An N×N×N 3D texture (any sampleable colour format — the app ships
    ///     `rgba16Float`). Non-cubic or non-3D textures are rejected. `nil` disables.
    ///   - strength: Blend weight toward the stock's rendering [0, 1]. Defaults 1.
    public func setFilmLUT(_ texture: MTLTexture?, strength: Float = 1.0) {
        guard let texture else {
            filmLUTTexture = nil
            filmLUTStrength = strength
            return
        }
        guard texture.textureType == .type3D,
              texture.width == texture.height, texture.height == texture.depth,
              texture.width >= 2 else {
            assertionFailure("film LUT must be a cubic 3D texture, got \(texture.textureType) \(texture.width)×\(texture.height)×\(texture.depth)")
            filmLUTTexture = nil
            return
        }
        filmLUTTexture = texture
        filmLUTStrength = strength
        filmLUTSize = Float(texture.width)
    }

    /// Diagnostic A/B override: `VIZ_ILLUMI_TAA=1`/`0` forces the TAA toggle for
    /// every Illuminatorama renderer in the process, bypassing both the panel and
    /// any scene `sharedTAAOverride`. Lets a headless render exercise the TAA path
    /// without UI. `nil` when the env var is unset. Read once.
    static let taaEnvOverride: Bool? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_TAA"] else { return nil }
        return v == "1" || v.lowercased() == "true"
    }()

    // Issue #65 — headless motion-blur override. Motion blur is otherwise driven
    // only by the panel, so a headless render can't exercise it; `VIZ_ILLUMI_MOTIONBLUR`
    // (a float, e.g. `1.0`) forces `motionBlurStrength` for every renderer in the
    // process so the velocity-buffer streak can be render-verified. `nil` when unset.
    nonisolated static let motionBlurEnvOverride: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_MOTIONBLUR"] else { return nil }
        return Float(v)
    }()

    // Issue #65 — headless contact-shadow override. Contact shadows are otherwise
    // driven only by the panel, so a headless render can't exercise them;
    // `VIZ_ILLUMI_CONTACTSHADOW` (a float strength, e.g. `0.85`) forces
    // `contactShadowStrength` for every renderer in the process so the contact
    // darkening can be render-verified A/B. `nil` when unset.
    nonisolated static let contactShadowEnvOverride: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_CONTACTSHADOW"] else { return nil }
        return Float(v)
    }()

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

    // ── Issue #65: colour-grading LUT ─────────────────────────────────────────

    /// IEEE-754 binary32 → binary16 (half) bit pattern for `rgba16Float` uploads.
    /// Software conversion: Swift's `Float16` is unavailable under Mac Catalyst.
    /// LUT values are finite and in [0, 1], so the round-to-nearest-even path is
    /// all that matters; sub-normals/inf/NaN are handled for completeness.
    @inline(__always)
    static func float32to16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let expo = Int32((bits >> 23) & 0xFF) - 127 + 15
        let mant = bits & 0x7F_FFFF
        if expo >= 0x1F {                                   // overflow / inf / NaN
            let isNaN = (bits & 0x7F80_0000) == 0x7F80_0000 && mant != 0
            return sign | (isNaN ? 0x7E00 : 0x7C00)
        }
        if expo <= 0 {                                      // sub-normal / zero
            guard expo > -11 else { return sign }
            let m = (mant | 0x80_0000) >> UInt32(1 - expo + 13)
            return sign | UInt16(m)
        }
        // Normal: round mantissa to 10 bits, nearest-even.
        var half = UInt32(expo) << 10 | (mant >> 13)
        let roundBits = mant & 0x1FFF
        if roundBits > 0x1000 || (roundBits == 0x1000 && (half & 1) == 1) { half += 1 }
        return sign | UInt16(truncatingIfNeeded: half)
    }

    /// Build an identity 3D LUT (`rgba16Float`, `size³`) where the texel at
    /// (r,g,b) holds the colour (r,g,b)/(size-1). Sampling it with a colour in
    /// [0,1]³ (half-texel-corrected) returns that colour unchanged — an exact
    /// no-op grade. Uploaded once; `.shared` storage so the CPU bake is visible.
    static func makeIdentityColorLUT(device: MTLDevice, size: Int) -> MTLTexture? {
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = size; d.height = size; d.depth = size
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        tex.label = "Illuminatorama.colorLUT"
        var data = [UInt16](repeating: 0, count: size * size * size * 4)
        let inv = 1.0 / Float(size - 1)
        var i = 0
        for b in 0..<size {
            for g in 0..<size {
                for r in 0..<size {
                    data[i+0] = float32to16(Float(r) * inv)
                    data[i+1] = float32to16(Float(g) * inv)
                    data[i+2] = float32to16(Float(b) * inv)
                    data[i+3] = float32to16(1.0)
                    i += 4
                }
            }
        }
        let bpr = size * 4 * MemoryLayout<UInt16>.size
        let bpi = bpr * size
        data.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size),
                        mipmapLevel: 0, slice: 0,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: bpr, bytesPerImage: bpi)
        }
        return tex
    }

    /// Replace the colour-grade LUT from packed RGBA float data laid out as the
    /// standard .cube ordering (red fastest, then green, then blue). `rgba` must
    /// hold `size³` SIMD4 entries. Pass `nil` to reset to identity. The grade only
    /// shows when `colorLUTAmount > 0`. Display-space (the LUT is sampled after
    /// tonemapping, on the sRGB-encoded colour).
    @discardableResult
    public func setColorLUT(size: Int, rgba: [SIMD4<Float>]?) -> Bool {
        guard let rgba else {
            if let id = Self.makeIdentityColorLUT(device: device, size: Self.colorLUTDefaultSize) {
                pfx.colorLUT = id; pfx.colorLUTSize = Self.colorLUTDefaultSize; return true
            }
            return false
        }
        guard size >= 2, rgba.count == size * size * size else { return false }
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = size; d.height = size; d.depth = size
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return false }
        tex.label = "Illuminatorama.colorLUT(custom)"
        var data = [UInt16](repeating: 0, count: rgba.count * 4)
        for (j, c) in rgba.enumerated() {
            data[j*4+0] = Self.float32to16(c.x)
            data[j*4+1] = Self.float32to16(c.y)
            data[j*4+2] = Self.float32to16(c.z)
            data[j*4+3] = Self.float32to16(1.0)
        }
        let bpr = size * 4 * MemoryLayout<UInt16>.size
        let bpi = bpr * size
        data.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size),
                        mipmapLevel: 0, slice: 0,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: bpr, bytesPerImage: bpi)
        }
        pfx.colorLUT = tex; pfx.colorLUTSize = size
        return true
    }

    /// Parse an Adobe/Resolve `.cube` 3D LUT and install it. Returns false on a
    /// malformed file. Supports `LUT_3D_SIZE` + `r g b` rows (the common subset).
    @discardableResult
    public func loadCubeLUT(contentsOf url: URL) -> Bool {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return false }
        var size = 0
        var entries: [SIMD4<Float>] = []
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            if line.uppercased().hasPrefix("LUT_3D_SIZE") {
                size = Int(line.split(separator: " ").last.map(String.init) ?? "") ?? 0
                if size > 1 { entries.reserveCapacity(size*size*size) }
                continue
            }
            let comps = line.split(separator: " ").compactMap { Float($0) }
            if comps.count == 3 { entries.append(SIMD4(comps[0], comps[1], comps[2], 1)) }
        }
        guard size > 1, entries.count == size*size*size else { return false }
        return setColorLUT(size: size, rgba: entries)
    }

    /// Generate one of the built-in procedural grades as packed RGBA LUT data
    /// (.cube order, red fastest). Returns nil for `.none` (caller installs the
    /// identity LUT). The grade is computed in DISPLAY (~sRGB) space, matching how
    /// the tonemap samples the LUT.
    static func generateColorGradeLUT(look: IlluminatoramaColorGradeLook,
                                      size: Int) -> [SIMD4<Float>]? {
        guard look != .none, size >= 2 else { return nil }
        func luma(_ c: SIMD3<Float>) -> Float { c.x*0.2126 + c.y*0.7152 + c.z*0.0722 }
        func contrast(_ c: SIMD3<Float>, _ k: Float) -> SIMD3<Float> { (c - 0.5) * k + 0.5 }
        func smooth(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
            let t = max(0, min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t)
        }
        var out = [SIMD4<Float>](); out.reserveCapacity(size*size*size)
        let inv = 1.0 / Float(size - 1)
        for b in 0..<size { for g in 0..<size { for r in 0..<size {
            var c = SIMD3(Float(r), Float(g), Float(b)) * inv
            switch look {
            case .none: break
            case .tealOrange:
                let t = smooth(0.2, 0.85, luma(c))
                let shadow = SIMD3<Float>(-0.02, 0.03, 0.08)   // push shadows teal
                let high   = SIMD3<Float>( 0.09, 0.02, -0.07)  // push highlights orange
                c += shadow * (1 - t) + high * t
                c = contrast(c, 1.10)
            case .warm:
                c *= SIMD3<Float>(1.07, 1.01, 0.90); c = contrast(c, 1.05)
            case .cool:
                c *= SIMD3<Float>(0.92, 1.00, 1.09); c = contrast(c, 1.05)
            case .noir:
                let l = luma(c); c = contrast(SIMD3(l, l, l), 1.22)
            case .vibrant:
                let l = luma(c); c = l + (c - l) * 1.45; c = contrast(c, 1.06)
            }
            c = simd_clamp(c, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
            out.append(SIMD4(c, 1))
        }}}
        return out
    }

    /// Apply the shared panel's colour grade (issue #65). Rebakes the LUT only when
    /// the chosen look changes (the CPU bake is a per-frame no-op in steady state),
    /// and sets `colorLUTAmount` each frame. Called from `render()` alongside the
    /// other shared-FX appliers, so every Illuminatorama scene inherits it.
    func applySharedColorGrade(_ s: IlluminatoramaSharedSettings = .shared) {
        if s.colorGradeLook != pfx.appliedColorGradeLook {
            pfx.appliedColorGradeLook = s.colorGradeLook
            let data = Self.generateColorGradeLUT(look: s.colorGradeLook,
                                                  size: Self.colorLUTDefaultSize)
            setColorLUT(size: Self.colorLUTDefaultSize, rgba: data)
        }
        colorLUTAmount = (s.colorGradeLook == .none) ? 0 : Float(s.colorLUTAmount)
    }

    /// Advance the eased Post-FX shadows toward their slider targets. `Instant`
    /// (tau = 0) snaps; otherwise exponential smoothing `k = 1 − e^(−dt/tau)`.
    /// Self-timed off `CACurrentMediaTime` so it needs no dt from the caller, and
    /// the first call (lastEaseTime == 0 → dt 0 → k 1) seeds the shadows at the
    /// current targets (no startup glide). Mirrors AspectTest's `easeCamera`.
    func advancePostFXEasing() {
        let now = CACurrentMediaTime()
        let dt = pfx.lastPostFXEaseTime == 0 ? 0 : max(0, min(0.1, fixedFrameDelta ?? (now - pfx.lastPostFXEaseTime)))
        pfx.lastPostFXEaseTime = now
        let k = (postFXEasingTau <= 0 || dt == 0) ? 1.0 : 1 - exp(-dt / postFXEasingTau)
        let kf = Float(k)
        pfx.easedExposure          += (exposure              - pfx.easedExposure)          * kf
        pfx.easedBloomThreshold    += (bloomThreshold        - pfx.easedBloomThreshold)    * kf
        pfx.easedBloomIntensity    += (bloomIntensity        - pfx.easedBloomIntensity)    * kf
        pfx.easedChromaticAberration += (max(0, chromaticAberration) - pfx.easedChromaticAberration) * kf
        pfx.easedFringe            += (max(0, fringe)        - pfx.easedFringe)            * kf
        pfx.easedFringeTint        += (fringeTint            - pfx.easedFringeTint)        * kf
        pfx.easedSphericalAberration += (max(0, sphericalAberration) - pfx.easedSphericalAberration) * kf
        pfx.easedHalationIntensity += (max(0, halationIntensity) - pfx.easedHalationIntensity) * kf
        pfx.easedHalationThreshold += (halationThreshold        - pfx.easedHalationThreshold) * kf
        pfx.easedHalationRadius    += (max(0, halationRadius)   - pfx.easedHalationRadius)    * kf
        pfx.easedHalationTint      += (halationTint             - pfx.easedHalationTint)      * kf
        // Exponential easing approaches 0 asymptotically, so a scene that turns halation
        // OFF would keep encoding three dispatches forever on a residue of ~1e-9. Snap
        // the last sliver: `encodeHalationPasses` and the shader branch both key off
        // this value, so the effect switches off exactly when the passes stop.
        if halationIntensity <= 0 && pfx.easedHalationIntensity < 1e-3 { pfx.easedHalationIntensity = 0 }
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

    // ── Bloom pyramid geometry (S1.2) ─────────────────────────────────────────

    /// Deepest level a bloom pyramid over an `halfW × halfH` mip0 may have.
    ///
    /// Two bounds, both about the filter rather than about memory. The 13-tap box
    /// reaches ±2 texels and the tent ±1, so a level whose SHORT side is under 8
    /// texels has clamped edge taps on most of its rows — past that the halo
    /// stops widening and starts smearing the frame border inward. And the reach
    /// doubles per level, so 7 levels is the cap: the coarsest mip is then a
    /// handful of texels covering the whole frame, a formless wash rather than a
    /// halo.
    ///
    /// **8, not 16, and that is a MEASURED number.** Each level restores one
    /// octave of the profile's self-similar ~0.45–0.5 falloff and the last level
    /// always falls off a cliff, so the pyramid's useful reach is one octave
    /// short of its depth. Cutting at a 16-texel short side gives 5 levels on a
    /// 1200 × 900 frame, and the gate measured exactly the predicted cliff there:
    /// octave ratios 0.53 · 0.52 · 0.44 · **0.21 · 0.02** — a halo dead by ~120
    /// output px, which cannot carry the "tail past 200 px" this pass exists to
    /// produce. Halving the bound buys the octave back.
    ///
    /// **DH-0998 — that rule is now the GLARE FIT's (`bloomGlareFitLevelCount`), not the pyramid's.**
    /// "The last level always falls off a cliff" was true of every depth: the coarsest level's kernel
    /// ends ~2^(levels+1.2) texels out, so wherever the pyramid stopped, the halo stopped — on the
    /// BloomProfile lamp (1200 × 900, 6 levels) the last octave measured Δ(128)/Δ(64) = 0.125 against
    /// ~0.5 for every octave before it. The pyramid now keeps halving until its coarsest level's
    /// reach (1.5·2^levels) covers the whole mip0 DIAGONAL (or the level is one texel), so no cliff
    /// is left inside the frame; the extra levels are a handful of texels each (≤ 15 × 9 at a hero
    /// still) and cost a few µs. The physical glare still fits only the levels this filter can
    /// realise cleanly — its far field is `illumi_bloom_glare_tail`, exact in angle, not more levels.
    static func bloomLevelCount(halfW: Int, halfH: Int) -> Int {
        let maxSide = max(1, max(halfW, halfH))
        let diagonal = Double(halfW * halfW + halfH * halfH).squareRoot()
        var levels = max(1, bloomGlareFitLevelCount(halfW: halfW, halfH: halfH))
        while 1.5 * pow(2, Double(levels)) < diagonal && (maxSide >> levels) >= 1 { levels += 1 }
        return levels
    }

    /// The levels the physical glare (DH-1001) is FITTED over: the pre-DH-0998 pyramid rule — short
    /// side ≥ 8 texels, at most 7 levels, the depth the GPU-measured kernel table describes without
    /// the frame border clamping into them. The pyramid may be deeper (`bloomLevelCount`); the fit
    /// gives the extra levels no weight.
    static func bloomGlareFitLevelCount(halfW: Int, halfH: Int) -> Int {
        let minSide = max(1, min(halfW, halfH))
        var levels = 1
        while levels < 7 && (minSide >> levels) >= 8 { levels += 1 }
        return levels
    }

    /// Pixel size of pyramid level `level` (0 = mip0 at internal/2). Floored at 1
    /// so an extreme aspect ratio can't ask for a zero-width texture.
    static func bloomLevelSize(halfW: Int, halfH: Int, level: Int) -> (Int, Int) {
        (max(1, halfW >> level), max(1, halfH >> level))
    }

    // Issue #65 — colour-grade LUT default per-axis resolution. 33 is the .cube
    // de-facto standard (matches DaVinci / most authoring tools).
    static let colorLUTDefaultSize: Int = 33
}
