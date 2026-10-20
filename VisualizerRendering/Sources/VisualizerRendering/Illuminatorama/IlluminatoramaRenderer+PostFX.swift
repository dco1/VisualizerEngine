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
        enc.setComputePipelineState(taaResolvePipeline)
        enc.setTexture(hdrCompositeTexture,     index: 0)
        enc.setTexture(previousHistoryTexture,  index: 1)
        enc.setTexture(velocityTexture,         index: 2)
        enc.setTexture(currentHistoryTexture,   index: 3)
        // Phase 4.44 — depth lets the resolve synthesise a camera-only motion
        // vector for sky pixels (which the G-buffer leaves at zero velocity).
        enc.setTexture(depthTexture,            index: 4)
        // Previous frame's depth — lets the resolve reject history at a
        // disocclusion (a fast mover vacated this pixel) even when the vacated
        // surface and the mover share luma, which neighbourhood clamping can't.
        enc.setTexture(previousDepthTexture,    index: 5)
        // The downstream frame, kept OUT of the accumulator at index 3 so the
        // resolve's acutance sharpen is applied once on the way out instead of
        // compounding inside the history every frame.
        enc.setTexture(taaResolvedTexture,      index: 6)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        var still = TAAStillParams(still: taaStillAccumulation ? 1 : 0, karis: taaStillKarisWeights ? 1 : 0,
                                   noSharpen: taaStillSharpen ? 0 : 1)
        enc.setBytes(&still, length: MemoryLayout<TAAStillParams>.stride, index: 1)
        dispatch(enc, pipeline: taaResolvePipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// What bloom + tonemap should read from this frame. With TAA on, that's
    /// the history texture we just wrote to; otherwise the SSR-composited HDR.
    private var postFXSource: MTLTexture {
        taaEnabled ? taaResolvedTexture : hdrCompositeTexture
    }

    /// The HDR colour post-FX (exposure, DOF, bloom, tonemap) should sample.
    /// After the post-resolve overlay it's the display buffer; otherwise the
    /// plain resolved source.
    private var displaySource: MTLTexture {
        overlayApplied ? hdrTexture : postFXSource
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
        overlayApplied = false
        let hasEmitters = particleEmitters.contains { $0.enabled }
        guard !particleFields.isEmpty || hasEmitters || !particleStreaks.isEmpty else { return }

        // Seed the display buffer with the resolved opaque image.
        let resolved = postFXSource
        if resolved !== hdrTexture {
            guard let blit = cb.makeBlitCommandEncoder() else { return }
            blit.label = "Illuminatorama.overlay.seed"
            blit.copy(from: resolved,
                      sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: width, height: height, depth: 1),
                      to: hdrTexture,
                      destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        }

        // Additive draws into the display buffer (all target `hdrTexture`).
        encodeParticleDraw(cb)
        encodeExternalPointDraw(cb)
        encodeStreakDraw(cb)           // Phase 4.44b — velocity-aligned streak quads
        overlayApplied = true
    }

    /// Source for bloom + tonemap: the DOF output when DOF ran this frame,
    /// otherwise the raw post-FX source. (Exposure estimate still reads the
    /// pre-DOF source so the blur doesn't perturb auto-exposure.)
    /// `hdrPreExposure`, sanitised (non-finite / ≤ 0 ⇒ 1).
    var preExposureDivisor: Float {
        hdrPreExposure.isFinite && hdrPreExposure > 0 ? hdrPreExposure : 1
    }

    var bloomTonemapSource: MTLTexture {
        localToneMapOutput ?? (dofApplied ? dofOutputTexture : nil) ?? displaySource
    }

    // ── Local tone mapping (IlluminatoramaLocalToneMap.swift) ─────────────
    /// Local adaptation of the finished optical image (post-DOF, post-outline) — what bloom,
    /// halation and the tonemap then read. No-op unless `localToneMapping.isEnabled`; the
    /// `VIZ_ILLUMI_HDR_DUMP_PATH` diagnostic dumps the frame it would adapt either way. The
    /// exposure it anchors on is the tonemap's own (`easedExposure` × the metered exposure).
    func encodeLocalToneMap(_ cb: MTLCommandBuffer) {
        let source = bloomTonemapSource
        if IlluminatoramaHDRDump.path != nil {
            hdrDump.encode(cb, source: source, exposureBuffer: exposureBuffer,
                           hostExposure: easedExposure / preExposureDivisor, autoExposure: autoExposureEnabled)
        }
        guard localToneMapping.isEnabled else { return }
        localToneMapOutput = localToneMapPass.encode(
            cb, source: source, exposureBuffer: exposureBuffer, hostExposure: easedExposure / preExposureDivisor,
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
        dofApplied = false
        if dofReferenceAccumulation {
            encodeDOFReferenceAccumulation(cb)
            return
        }
        guard dofEnabled, dofCoCCoefficient > 0,
              let pipeline = dofPipeline, let tilePipeline = dofTilePipeline,
              let dilatePipeline = dofDilatePipeline else { return }
        // Lazily (re)allocate the DOF target to the current internal size.
        if dofOutputTexture == nil
            || dofOutputTexture?.width != width || dofOutputTexture?.height != height {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = .rgba16Float
            d.width = max(1, width); d.height = max(1, height)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofOutputTexture = device.makeTexture(descriptor: d)
            dofOutputTexture?.label = "Illuminatorama.dof"
        }
        let tile = Self.dofTileSize
        let tileW = max(1, (width + tile - 1) / tile), tileH = max(1, (height + tile - 1) / tile)
        if dofTileTexture == nil
            || dofTileTexture?.width != tileW || dofTileTexture?.height != tileH {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            // r = max |CoC|, g = the near field's max CoC (the fast gather's bound; unused otherwise).
            d.pixelFormat = .rg32Float
            d.width = tileW; d.height = tileH
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofTileTexture = device.makeTexture(descriptor: d)
            dofTileTexture?.label = "Illuminatorama.dof.cocTiles"
            dofTileDilatedTexture = device.makeTexture(descriptor: d)
            dofTileDilatedTexture?.label = "Illuminatorama.dof.cocTilesDilated"
        }
        // The prefilter pyramid: half resolution, full mip chain. Sized from the internal
        // render size like everything else here, so it follows the draft/settled scale.
        let pw = max(1, (width + 1) / 2), ph = max(1, (height + 1) / 2)
        // The min-CoC pyramid's format: R = least |CoC| (the sharp-subject tests) everywhere; the
        // layered composite (`dofPartialOcclusion`) also reads G/B = the signed-CoC range, so only
        // it pays for an rgba16Float pyramid.
        let minCoCFormat: MTLPixelFormat = dofSharpSubjects && dofPartialOcclusion ? .rgba16Float : .r16Float
        if dofPrefilterTexture == nil
            || dofPrefilterTexture?.width != pw || dofPrefilterTexture?.height != ph
            || dofMinCoCTexture?.pixelFormat != minCoCFormat {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = .rgba16Float
            d.width = pw; d.height = ph
            d.mipmapLevelCount = Int(log2(Double(max(pw, ph))).rounded(.down)) + 1
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofPrefilterTexture = device.makeTexture(descriptor: d)
            dofPrefilterTexture?.label = "Illuminatorama.dof.prefilter"
            // The min-CoC pyramid rides along (always allocated with it, so the kernels' slots are
            // always bound; only `dofSharpSubjects` writes or reads it).
            d.pixelFormat = minCoCFormat
            dofMinCoCTexture = device.makeTexture(descriptor: d)
            dofMinCoCTexture?.label = "Illuminatorama.dof.minCoC"
            dofMinCoCLevels = (0..<d.mipmapLevelCount).compactMap { l in
                dofMinCoCTexture?.makeTextureView(pixelFormat: minCoCFormat, textureType: .type2D,
                                                  levels: l..<(l + 1), slices: 0..<1)
            }
            dofNearTexture = nil
            dofNearCoCTexture = nil
        }
        // The layered composite's near-field coverage pyramid (the prefilter's size and mip chain).
        if dofSharpSubjects && dofPartialOcclusion, dofNearTexture == nil,
           let pre = dofPrefilterTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = .rgba16Float
            d.width = pre.width; d.height = pre.height
            d.mipmapLevelCount = pre.mipmapLevelCount
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofNearTexture = device.makeTexture(descriptor: d)
            dofNearTexture?.label = "Illuminatorama.dof.nearField"
            d.pixelFormat = .r16Float
            dofNearCoCTexture = device.makeTexture(descriptor: d)
            dofNearCoCTexture?.label = "Illuminatorama.dof.nearCoC"
        }
        guard let out = dofOutputTexture, let tiles = dofTileTexture,
              let dilated = dofTileDilatedTexture else { return }
        // Point taps unless the pyramid is genuinely there to read: the shader treats 0 as
        // "sample the frame directly", which is the correct fallback and NOT a silent one —
        // it is the only value the field ever held before DH-0883.
        let prefilter: MTLTexture? = dofPrefilterPipeline != nil ? dofPrefilterTexture : nil
        // The sharp-subject gather needs its min-CoC pyramid (and the prefilter that seeds it).
        let minCoC = dofMinCoCTexture
        let sharpSubjects = dofSharpSubjects && prefilter != nil && minCoC != nil && dofMinCoCPipeline != nil
            && dofMinCoCLevels.count == (minCoC?.mipmapLevelCount ?? -1)
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
        let nearOn = p.partialOcclusion > 0.5 && dofNearTexture != nil && dofNearCoCTexture != nil
        p.nearPyramid = nearOn ? 1 : 0
        // The near-field slots: the pyramid while it is on, else any texture of the pyramid's size
        // (bound, never read or written — the kernels test `nearPyramid`).
        let nearBinding: MTLTexture? = nearOn ? dofNearTexture : prefilter
        let nearCoCBinding: MTLTexture? = nearOn ? dofNearCoCTexture : minCoC
        // The fast gather needs the half-res target (same size as the pyramid's base).
        if sharpSubjects && dofFastGather && dofHalfPipeline != nil,
           dofHalfTexture == nil || dofHalfTexture?.width != pw || dofHalfTexture?.height != ph {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: pw, height: ph,
                                                             mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofHalfTexture = device.makeTexture(descriptor: d)
            dofHalfTexture?.label = "Illuminatorama.dof.half"
        }
        let fast = sharpSubjects && dofFastGather && dofHalfPipeline != nil && dofHalfTexture != nil
        p.fastGather = fast ? 1 : 0
        p.halfMinCoC = max(1, dofHalfResMinCoC)
        // The quarter tier (opt-in): its target is a quarter of the internal size (the prefilter
        // pyramid's level 1), and it needs the pyramid's mips for the blocks' own colour.
        let qw = max(1, (width + 3) / 4), qh = max(1, (height + 3) / 4)
        if fast && dofQuarterResMinCoC > 0 && dofQuarterPipeline != nil,
           dofQuarterTexture == nil || dofQuarterTexture?.width != qw || dofQuarterTexture?.height != qh {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: qw, height: qh,
                                                             mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofQuarterTexture = device.makeTexture(descriptor: d)
            dofQuarterTexture?.label = "Illuminatorama.dof.quarter"
        }
        let quarter = fast && dofQuarterResMinCoC > 0 && dofQuarterPipeline != nil && dofQuarterTexture != nil
        p.quarterMinCoC = quarter ? max(6, dofQuarterResMinCoC) : 0
        guard let tileEnc = timedComputeEncoder(cb, "dof.tiles") else { return }
        tileEnc.label = "Illuminatorama.dof.tiles"
        tileEnc.setTexture(depthTexture, index: 0)
        tileEnc.setTexture(tiles, index: 1)
        tileEnc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        if dofFastGather, let par = dofTileParallelPipeline {
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
        if let prePipeline = dofPrefilterPipeline, let pre = prefilter {
            if let preEnc = timedComputeEncoder(cb, "dof.prefilter") {
                preEnc.label = "Illuminatorama.dof.prefilter"
                preEnc.setComputePipelineState(prePipeline)
                preEnc.setTexture(displaySource, index: 0)
                preEnc.setTexture(depthTexture, index: 1)
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
                if nearOn, let near = dofNearTexture, let nearCoC = dofNearCoCTexture {
                    blit.generateMipmaps(for: near)
                    blit.generateMipmaps(for: nearCoC)
                }
                blit.endEncoding()
            }
            // Min-CoC levels 1… from the prefilter-written base (sharp-subject mode only).
            if sharpSubjects, let minP = dofMinCoCPipeline, dofMinCoCLevels.count > 1,
               let mEnc = timedComputeEncoder(cb, "dof.minCoC") {
                mEnc.label = "Illuminatorama.dof.minCoC"
                mEnc.setComputePipelineState(minP)
                for l in 1..<dofMinCoCLevels.count {
                    let dst = dofMinCoCLevels[l]
                    mEnc.setTexture(dofMinCoCLevels[l - 1], index: 0)
                    mEnc.setTexture(dst, index: 1)
                    dispatch(mEnc, pipeline: minP, width: dst.width, height: dst.height)
                }
                mEnc.endEncoding()
            }
        }

        // The quarter tier first (opt-in): the half-res pass reads it to skip the texels it serves.
        if quarter, let quarterP = dofQuarterPipeline, let qTex = dofQuarterTexture,
           let qEnc = timedComputeEncoder(cb, "dof.quarter") {
            qEnc.label = "Illuminatorama.dof.quarter"
            qEnc.setComputePipelineState(quarterP)
            qEnc.setTexture(displaySource, index: 0)
            qEnc.setTexture(depthTexture, index: 1)
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
        let quarterBinding: MTLTexture? = quarter ? dofQuarterTexture : prefilter

        if fast, let halfP = dofHalfPipeline, let half = dofHalfTexture,
           let hEnc = timedComputeEncoder(cb, "dof.half") {
            hEnc.label = "Illuminatorama.dof.half"
            hEnc.setComputePipelineState(halfP)
            hEnc.setTexture(displaySource, index: 0)
            hEnc.setTexture(depthTexture, index: 1)
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
        enc.setTexture(depthTexture, index: 1)
        enc.setTexture(out, index: 2)
        enc.setTexture(dilated, index: 3)
        enc.setTexture(prefilter, index: 4)
        enc.setTexture(minCoC, index: 5)
        // The fast gather's half-res result (any half-size texture when it is off: bound, unread).
        enc.setTexture(fast ? dofHalfTexture : prefilter, index: 6)
        enc.setTexture(quarterBinding, index: 7)
        enc.setTexture(nearBinding, index: 8)
        enc.setTexture(nearCoCBinding, index: 9)
        enc.setBytes(&p, length: MemoryLayout<DOFParams>.stride, index: 0)
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
        dofApplied = true
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
        if dofOutputTexture == nil
            || dofOutputTexture?.width != width || dofOutputTexture?.height != height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: max(1, width),
                                                             height: max(1, height), mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofOutputTexture = device.makeTexture(descriptor: d)
            dofOutputTexture?.label = "Illuminatorama.dof"
        }
        if dofReferenceAccumTexture == nil
            || dofReferenceAccumTexture?.width != width || dofReferenceAccumTexture?.height != height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: max(1, width),
                                                             height: max(1, height), mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            dofReferenceAccumTexture = device.makeTexture(descriptor: d)
            dofReferenceAccumTexture?.label = "Illuminatorama.dof.reference"
            dofReferenceNeedsReset = true
        }
        guard let out = dofOutputTexture, let accum = dofReferenceAccumTexture,
              let enc = timedComputeEncoder(cb, "dof.reference") else { return }
        let reset = dofReferenceNeedsReset
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
        if reset { dofReferenceFrameCount = 0; dofReferenceNeedsReset = false }
        if add { dofReferenceFrameCount += 1 }
        dofApplied = true
    }

    func encodeExposureEstimate(_ cb: MTLCommandBuffer) {
        guard autoExposureEnabled else { return }
        // Held (opt-in): the smoothed exposure in `exposureBuffer` stays exactly where it is.
        guard !autoExposureHold else { return }
        // Per-frame dt, clamped to a sane window so a stalled frame
        // (debugger pause, scene reload) doesn't pump the EMA.
        let now = CACurrentMediaTime()
        lastFrameDuration = max(0.001, min(0.5, fixedFrameDelta ?? (now - lastExposureTickTime)))
        lastExposureTickTime = now
        guard let enc = timedComputeEncoder(cb, "exposureEstimate") else { return }
        enc.label = "Illuminatorama.exposureEstimate"
        enc.setComputePipelineState(exposureEstimatePipeline)
        enc.setTexture(displaySource, index: 0)
        enc.setBuffer(exposureBuffer, offset: 0, index: 0)
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
        enc.setBuffer(exposureHistogramBuffer, offset: 0, index: 5)
        // Update the host-driven `dt` slot in the buffer — the EMA step
        // inside the kernel needs to know how much wall-time elapsed
        // since the last estimate so the half-life math is correct.
        let dt = Float(max(0.001, lastFrameDuration))
        // The slot is 4 floats in; ExposureState.deltaTime is offset 12.
        let dtPtr = exposureBuffer.contents().advanced(by: 12)
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
        bloomUpChain.first ?? bloomDownChain[0]
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
        let mip0 = bloomDownChain[0]
        enc.setComputePipelineState(bloomPrefilterPipeline)
        enc.setTexture(bloomTonemapSource, index: 0)
        enc.setTexture(mip0, index: 1)
        dispatch(enc, pipeline: bloomPrefilterPipeline, width: mip0.width, height: mip0.height)

        // Down chain — each level is a 13-tap box of the one above it.
        enc.setComputePipelineState(bloomDownPipeline)
        for level in 1..<bloomDownChain.count {
            let dst = bloomDownChain[level]
            enc.setTexture(bloomDownChain[level - 1], index: 0)
            enc.setTexture(dst, index: 1)
            dispatch(enc, pipeline: bloomDownPipeline, width: dst.width, height: dst.height)
        }

        // DH-0998 — the physical glare's far field: an exact angular convolution over one coarse
        // down level (see `illumi_bloom_glare_tail`), folded into the finest up step below.
        var tail: MTLTexture?
        var tailScale: Float = 0
        if let fit = lastBloomGlareFit, fit.energy > 0, !bloomUpChain.isEmpty,
           case let src = bloomDownChain[min(fit.tailGridLevel, bloomDownChain.count - 1)],
           let out = glareTailOutput(width: max(1, src.width / 2), height: max(1, src.height / 2)) {
            encodeGlareTail(enc, fit: fit, tanHalf: glareTailTanHalf, source: src, output: out)
            enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
            tail = out
            tailScale = Float(1 / fit.energy)
        }

        // Up chain — coarsest to finest. The first step is seeded from the
        // smallest DOWN level (which is why the up chain is one texture shorter).
        enc.setComputePipelineState(bloomUpPipeline)
        let last = bloomDownChain.count - 1
        let levelScatters = lastBloomGlareFit.map { fit in
            (0..<bloomDownChain.count).map { $0 < fit.scatters.count ? fit.scatters[$0] : 1 }
        } ?? Self.bloomPSFScatters(levels: bloomDownChain.count, exponent: bloomPSFExponent)
        for level in stride(from: last - 1, through: 0, by: -1) {
            let low = (level == last - 1) ? bloomDownChain[last] : bloomUpChain[level + 1]
            let dst = bloomUpChain[level]
            enc.setTexture(low, index: 0)
            enc.setTexture(bloomDownChain[level], index: 1)
            enc.setTexture(dst, index: 2)
            var s: Float = levelScatters?[level] ?? -1
            enc.setBytes(&s, length: MemoryLayout<Float>.stride, index: 1)
            var ts: Float = level == 0 ? tailScale : 0
            enc.setTexture(level == 0 ? (tail ?? low) : low, index: 3)
            enc.setBytes(&ts, length: MemoryLayout<Float>.stride, index: 2)
            dispatch(enc, pipeline: bloomUpPipeline, width: dst.width, height: dst.height)
        }
        enc.endEncoding()
    }

    /// The tail's output texture at `width × height` (rgba32Float — it is tiny), reused per size.
    /// Half the source grid in each axis (see `IlluminatoramaGlareSpread.Fit.tailGridLevel`).
    private func glareTailOutput(width: Int, height: Int) -> MTLTexture? {
        if let t = glareTailTexture, t.width == width, t.height == height { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: width, height: height,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        glareTailTexture = device.makeTexture(descriptor: d)
        glareTailTexture?.label = "Illuminatorama.bloom.glareTail"
        return glareTailTexture
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
        enc.setComputePipelineState(bloomPrefilterPipeline)
        enc.setTexture(input, index: 0); enc.setTexture(down[0], index: 1)
        dispatch(enc, pipeline: bloomPrefilterPipeline, width: down[0].width, height: down[0].height)
        enc.setComputePipelineState(bloomDownPipeline)
        for level in stride(from: 1, to: levels, by: 1) {
            enc.setTexture(down[level - 1], index: 0); enc.setTexture(down[level], index: 1)
            dispatch(enc, pipeline: bloomDownPipeline, width: down[level].width, height: down[level].height)
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
            enc.setComputePipelineState(bloomUpPipeline)
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
                dispatch(enc, pipeline: bloomUpPipeline, width: up[level].width, height: up[level].height)
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
        guard easedHalationIntensity > 0 else { return }
        let halfW = halationBright.width
        let halfH = halationBright.height
        guard let enc = timedComputeEncoder(cb, "halation") else { return }
        enc.label = "Illuminatorama.halation"

        enc.setComputePipelineState(halationThresholdPipeline)
        enc.setTexture(bloomTonemapSource, index: 0)
        enc.setTexture(halationBright, index: 1)
        enc.setBuffer(frameUniformBuffer, offset: 0, index: 0)
        dispatch(enc, pipeline: halationThresholdPipeline, width: halfW, height: halfH)

        enc.setComputePipelineState(halationBlurHPipeline)
        enc.setTexture(halationBright, index: 0)
        enc.setTexture(halationBlurH, index: 1)
        dispatch(enc, pipeline: halationBlurHPipeline, width: halfW, height: halfH)

        enc.setComputePipelineState(halationBlurVPipeline)
        enc.setTexture(halationBlurH, index: 0)
        enc.setTexture(halationBlurV, index: 1)
        dispatch(enc, pipeline: halationBlurVPipeline, width: halfW, height: halfH)
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
        pass.colorAttachments[0].texture = tonemapWriteTarget
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let enc = timedRenderEncoder(cb, pass, "tonemap") else { return }
        enc.label = "Illuminatorama.tonemap"
        enc.setRenderPipelineState(tonemapPipeline)
        enc.setFragmentTexture(bloomTonemapSource, index: 0)
        enc.setFragmentTexture(bloomResultTexture, index: 1)
        // Issue #65 — 3D colour-grade LUT at texture(2). Always bound (identity by
        // default), so the shader never branches on nil; the grade is gated by
        // `colorLUTAmount` in the uniforms.
        enc.setFragmentTexture(colorLUT, index: 2)
        // Issue #65 — screen-space velocity at texture(3) for motion blur. Always
        // bound; gated by `motionBlurStrength` (0 = the tonemap skips the gather).
        enc.setFragmentTexture(velocityTexture, index: 3)
        // Issue #65 — G-buffer channels at texture(4..6): albedo / normal /
        // roughness / metalness / depth. Always bound. TWO consumers now read
        // them — the Debug-view readouts (`debugTerm >= 10`) and the diagram
        // look (`diagramParams.x > 0`), which builds its flat fills and its
        // outlines out of exactly these three. Both are gated on their own
        // uniform, so a default photoreal render still samples neither.
        enc.setFragmentTexture(gbufferAlbedoMet, index: 4)
        enc.setFragmentTexture(gbufferNormalRgh, index: 5)
        enc.setFragmentTexture(depthTexture, index: 6)
        // Phase 9 — film LUT at texture(7); 2..6 were taken by the colour-grade
        // LUT / velocity / G-buffer binds in the merge. nil = disabled (the shader
        // checks `frame.filmLUTStrength > 0` before sampling).
        enc.setFragmentTexture(filmLUTTexture, index: 7)
        // Halation halo at texture(8). Always bound (the texture is allocated
        // unconditionally); its contents are stale when halationIntensity == 0,
        // and the shader's branch is gated on the same uniform, so it isn't read then.
        enc.setFragmentTexture(halationBlurV, index: 8)
        // The AO field at texture(9) for `DebugTerm.ssao` — the same texture the
        // lighting pass multiplies its indirect terms by (post-denoise when the
        // denoiser is on), so the debug view IS the shipped multiplier, not a
        // re-derivation. Always bound; only read when `debugTerm == 16`.
        enc.setFragmentTexture(aoSourceTexture, index: 9)
        enc.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 0)
        // Phase 4.21 — exposure buffer at buffer(1). The fragment reads
        // `expoState.smoothedExposure` when `frame.autoExposureEnabled`
        // is set, falls back to `frame.exposure` otherwise.
        enc.setFragmentBuffer(exposureBuffer, offset: 0, index: 1)
        // Fullscreen triangle — 3 vertices, positions synthesised in the
        // vertex shader from `[[vertex_id]]`; no vertex buffer needed.
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }
}
