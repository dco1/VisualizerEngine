import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
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
    var bloomResultTexture: MTLTexture {
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
}
