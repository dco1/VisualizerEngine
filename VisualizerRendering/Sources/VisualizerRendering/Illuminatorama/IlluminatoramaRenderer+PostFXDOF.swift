import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
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
}
