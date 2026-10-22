import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    // ── Target allocation ─────────────────────────────────────────────────────

    struct Targets {
        var albedoMet: MTLTexture
        var normalRgh: MTLTexture
        var emission: MTLTexture
        var depth: MTLTexture
        var hdr: MTLTexture
        var ao: MTLTexture
        var hdrComposite: MTLTexture
        var rtDiffuse: MTLTexture    // full-res RT diffuse (shadow+GI) pre-denoise
        // Phase 2.7
        var velocity: MTLTexture
        // Light-layer G-buffer target (R32Uint). Carries each fragment's `layer`
        // bitfield so the deferred lighting kernel can mask per-room lights. Cleared
        // to 0xFFFFFFFF (sky/no-geometry) — the all-bits default means no scene that
        // leaves layers/masks untouched sees any change.
        var layer: MTLTexture
        var historyA: MTLTexture
        var historyB: MTLTexture
        /// The TAA resolve's DOWNSTREAM output — the sharpened frame bloom and
        /// tonemap read. Separate from the ping-pong pair because those two are
        /// the ACCUMULATOR, and the acutance sharpen must not compound inside it.
        var taaResolved: MTLTexture
        // Bloom mip pyramid (S1.2) — see `bloomDownChain` for why these are
        // separate per-level textures rather than one mipmapped resource.
        var bloomDown: [MTLTexture]      // level 0 = internal/2, each halving
        var bloomUp: [MTLTexture]        // one SHORTER than `bloomDown`
        // Film halation — half-res (see `halationBright`).
        var halationBright: MTLTexture
        var halationBlurH: MTLTexture
        var halationBlurV: MTLTexture
        var ldr: MTLTexture
        // Phase 4.39 denoiser
        var aoFiltered: MTLTexture   // half-res, after bilateral
        var aoHistoryA: MTLTexture   // half-res r16Float ping-pong
        var aoHistoryB: MTLTexture
        var ssrRaw: MTLTexture       // full-res SSR gather delta
        var ssrHistoryA: MTLTexture  // full-res rgba16Float ping-pong
        var ssrHistoryB: MTLTexture
        var rtGIHistoryA: MTLTexture // full-res rgba16Float ping-pong (RT GI temporal)
        var rtGIHistoryB: MTLTexture
        // SVGF denoiser (Phase 4.44)
        var aoSampleCount: MTLTexture    // half-res r16Float — accumulated frame count per SSAO pixel
        var ssrSampleCount: MTLTexture   // full-res r16Float — accumulated frame count per SSR pixel
        var giSampleCount: MTLTexture    // full-res r16Float — accumulated frame count per RT GI pixel
        var giVariance: MTLTexture       // full-res r16Float — variance ping-pong A
        var giVarianceB: MTLTexture      // full-res r16Float — variance ping-pong B
        var giAtrousA: MTLTexture        // full-res rgba16Float — à-trous ping-pong A
        var giAtrousB: MTLTexture        // full-res rgba16Float — à-trous ping-pong B
        // Issue #65 — screen-space SSS. `sssDiffuse` holds the diffuse-lit term the
        // lighting pass peels off SSS-flagged pixels (rgb) + the mask (a); the blur
        // ping-pongs sssDiffuse → sssBlurTmp (H) → sssBlurOut (V); the composite
        // reads sssDiffuse (sharp) + sssBlurOut (blurred). Only written/read when a
        // scene sets sssStrength > 0; otherwise allocated-but-idle.
        var sssDiffuse: MTLTexture       // full-res rgba16Float — diffuse + mask
        var sssBlurTmp: MTLTexture       // full-res rgba16Float — horizontal-blur intermediate
        var sssBlurOut: MTLTexture       // full-res rgba16Float — separable-blur result
    }

    /// Standalone sibling of the G-buffer depth, sized to match it, used as the
    /// PREVIOUS frame's depth in the TAA resolve's disocclusion test. Not part of
    /// the texture pool — it's blit-filled from `depthTexture` at the end of each
    /// frame. `.shaderRead` so the resolve kernel can `read()` it; `.renderTarget`
    /// only to mirror the source depth's descriptor.
    static func makeDepthLike(device: MTLDevice, _ tex: MTLTexture) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: tex.width, height: tex.height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            throw IlluminatoramaError.bufferAllocationFailed("Illuminatorama.depth.previous")
        }
        t.label = "Illuminatorama.depth.previous"
        return t
    }

    /// `illumi_fs` specialised for one lane (DH-0140): `extended` selects the six-target output
    /// struct. The fragment carries a function constant now, so it can no longer be made by name alone.
    static func makeGBufferFragment(library: MTLLibrary, extended: Bool, interior: Bool = false) throws -> MTLFunction {
        let c = MTLFunctionConstantValues()
        var ext = extended
        c.setConstantValue(&ext, type: .bool, index: 11)
        var imap = interior
        c.setConstantValue(&imap, type: .bool, index: 12)
        return try library.makeFunction(name: "illumi_fs", constantValues: c)
    }

    /// The interior-mapping G-buffer pipeline for this lane (built once; nil if it fails to compile,
    /// in which case the kind draws with the ordinary pipeline — its flat emission).
    func interiorGBufferPipeline(extended: Bool) -> MTLRenderPipelineState? {
        if let p = extended ? gbs.gbufferPipelineInteriorExt : gbs.gbufferPipelineInterior { return p }
        guard let library = engine.library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = extended ? "Illuminatorama.gbuffer.interior.ext" : "Illuminatorama.gbuffer.interior"
        let vsC = MTLFunctionConstantValues()
        var pv = false
        vsC.setConstantValue(&pv, type: .bool, index: 10)
        do {
            d.vertexFunction = try library.makeFunction(name: "illumi_vs", constantValues: vsC)
            d.fragmentFunction = try Self.makeGBufferFragment(library: library, extended: extended, interior: true)
        } catch {
            Self.log.error("interior-mapping G-buffer: compile failed: \(error.localizedDescription)")
            return nil
        }
        d.colorAttachments[0].pixelFormat = .rgba16Float
        d.colorAttachments[1].pixelFormat = .rgba16Float
        d.colorAttachments[2].pixelFormat = .rgba16Float
        d.colorAttachments[3].pixelFormat = .rg16Float
        d.colorAttachments[4].pixelFormat = .r32Uint
        if extended { d.colorAttachments[5].pixelFormat = .rgba16Float }
        d.depthAttachmentPixelFormat = .depth32Float
        Self.declareVisBufferAttachment(d, device: device)
        let p = try? device.makeRenderPipelineState(descriptor: d)
        if extended { gbs.gbufferPipelineInteriorExt = p } else { gbs.gbufferPipelineInterior = p }
        return p
    }

    /// DH-0140 — build one six-target G-buffer pipeline variant. Same attachments as the live PSO
    /// plus color(5) RGBA16F for `GBufferOut.material`.
    static func makeExtendedGBufferPipeline(device: MTLDevice, library: MTLLibrary,
                                                    prevVerts: Bool) throws -> MTLRenderPipelineState {
        let d = MTLRenderPipelineDescriptor()
        d.label = prevVerts ? "Illuminatorama.gbuffer.prevVerts.ext" : "Illuminatorama.gbuffer.ext"
        let vsC = MTLFunctionConstantValues()
        var pv = prevVerts
        vsC.setConstantValue(&pv, type: .bool, index: 10)
        d.vertexFunction = try library.makeFunction(name: "illumi_vs", constantValues: vsC)
        d.fragmentFunction = try makeGBufferFragment(library: library, extended: true)
        d.colorAttachments[0].pixelFormat = .rgba16Float
        d.colorAttachments[1].pixelFormat = .rgba16Float
        d.colorAttachments[2].pixelFormat = .rgba16Float
        d.colorAttachments[3].pixelFormat = .rg16Float
        d.colorAttachments[4].pixelFormat = .r32Uint
        d.colorAttachments[5].pixelFormat = .rgba16Float
        d.depthAttachmentPixelFormat = .depth32Float
        declareVisBufferAttachment(d, device: device)
        return try device.makeRenderPipelineState(descriptor: d)
    }

    /// DH-0140 — make the sixth target and the two extended pipelines exist for this frame, at the
    /// current internal size. Returns false (logged) if Metal refuses, in which case the frame runs
    /// the five-target pass and the lighting kernel stays on its live-lane constants.
    func ensureExtendedGBufferResources() -> Bool {
        if gbs.gbufferPipelineExt == nil || gbs.gbufferPipelinePrevVertsExt == nil {
            guard let library = engine.library else {
                Self.log.error("extended G-buffer: no shader library — staying on the 5-target pass")
                return false
            }
            do {
                gbs.gbufferPipelineExt = try Self.makeExtendedGBufferPipeline(device: device, library: library, prevVerts: false)
                gbs.gbufferPipelinePrevVertsExt = try Self.makeExtendedGBufferPipeline(device: device, library: library, prevVerts: true)
            } catch {
                Self.log.error("extended G-buffer: pipeline compile failed: \(error.localizedDescription) — staying on the 5-target pass")
                return false
            }
        }
        if let t = gbs.gbufferMaterial, t.width == width, t.height == height { return true }
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        d.pixelFormat = .rgba16Float
        d.width = width
        d.height = height
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            Self.log.error("extended G-buffer: material target allocation failed at \(self.width)×\(self.height)")
            return false
        }
        t.label = "Illuminatorama.gbuffer.material"
        gbs.gbufferMaterial = t
        return true
    }

    static func makeTargets(
        device: MTLDevice,
        internalW: Int, internalH: Int,
        outputW: Int, outputH: Int
    ) throws -> Targets {
        func make(label: String, format: MTLPixelFormat, w: Int, h: Int,
                  usage: MTLTextureUsage) throws -> MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = .type2D
            d.pixelFormat = format
            d.width = w
            d.height = h
            d.usage = usage
            d.storageMode = .private
            guard let t = device.makeTexture(descriptor: d) else {
                throw IlluminatoramaError.bufferAllocationFailed(label)
            }
            t.label = label
            return t
        }
        // All internal-pipeline targets are sized to internalW × internalH —
        // these are the textures every compute kernel except the tonemap
        // reads from and writes into. The tonemap's `outputTexture` is the
        // only thing at the smaller output size.
        let am = try make(label: "Illuminatorama.gbuffer.albedoMet",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let nr = try make(label: "Illuminatorama.gbuffer.normalRgh",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let em = try make(label: "Illuminatorama.gbuffer.emission",
                          format: .rgba16Float, w: internalW, h: internalH,
                          usage: [.renderTarget, .shaderRead])
        let depth = try make(label: "Illuminatorama.depth",
                             format: .depth32Float, w: internalW, h: internalH,
                             usage: [.renderTarget, .shaderRead])
        // HDR composite and history need .sample because the tonemap kernel
        // bilinearly downsamples them into the output texture.
        let hdr = try make(label: "Illuminatorama.hdr",
                           format: .rgba16Float, w: internalW, h: internalH,
                           usage: [.shaderRead, .shaderWrite, .renderTarget])
        // Phase 4.11 — `.renderTarget` enables the additive forward
        // particles pass to draw directly into the composite. Compute
        // passes elsewhere still bind it via shaderRead/Write.
        let hdrComposite = try make(label: "Illuminatorama.hdrComposite",
                                    format: .rgba16Float, w: internalW, h: internalH,
                                    usage: [.renderTarget, .shaderRead, .shaderWrite])
        // RT diffuse (soft shadow + 1-bounce GI) buffer, full-res. The RT pass
        // writes here; `illumi_rt_denoise` bilateral-filters it into the composite.
        let rtDiffuse = try make(label: "Illuminatorama.rt.diffuse",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let velocity = try make(label: "Illuminatorama.velocity",
                                format: .rg16Float, w: internalW, h: internalH,
                                usage: [.renderTarget, .shaderRead])
        // Light-layer target — R32Uint so the full 32-bit bitfield round-trips
        // exactly (no half-float rounding). Written by the G-buffer fragment
        // shaders, read by the lighting kernel at [[texture(18)]].
        let layer = try make(label: "Illuminatorama.gbuffer.layer",
                             format: .r32Uint, w: internalW, h: internalH,
                             usage: [.renderTarget, .shaderRead])
        let historyA = try make(label: "Illuminatorama.historyA",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let historyB = try make(label: "Illuminatorama.historyB",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let taaResolved = try make(label: "Illuminatorama.taaResolved",
                                   format: .rgba16Float, w: internalW, h: internalH,
                                   usage: [.shaderRead, .shaderWrite])
        // AO and bloom run at half of the INTERNAL resolution (still oversampled
        // vs the output when SSAA is on).
        let halfW = max(1, internalW / 2)
        let halfH = max(1, internalH / 2)
        let ao = try make(label: "Illuminatorama.ao",
                          format: .r16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        // Phase 4.39 — SSAO denoiser intermediates.
        let aoFiltered = try make(label: "Illuminatorama.ao.filtered",
                                  format: .r16Float, w: halfW, h: halfH,
                                  usage: [.shaderRead, .shaderWrite])
        let aoHistA = try make(label: "Illuminatorama.ao.historyA",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        let aoHistB = try make(label: "Illuminatorama.ao.historyB",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        // Phase 4.39 — SSR denoiser intermediates.
        let ssrRaw = try make(label: "Illuminatorama.ssr.raw",
                              format: .rgba16Float, w: internalW, h: internalH,
                              usage: [.shaderRead, .shaderWrite])
        let ssrHistA = try make(label: "Illuminatorama.ssr.historyA",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let ssrHistB = try make(label: "Illuminatorama.ssr.historyB",
                                format: .rgba16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        // RT-GI temporal accumulation history (ping-pong, full-res).
        let rtGiHistA = try make(label: "Illuminatorama.rt.giHistoryA",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let rtGiHistB = try make(label: "Illuminatorama.rt.giHistoryB",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        // Phase 4.44 — SVGF denoiser: adaptive sample counts + à-trous buffers.
        let aoCount = try make(label: "Illuminatorama.svgf.aoCount",
                               format: .r16Float, w: halfW, h: halfH,
                               usage: [.shaderRead, .shaderWrite])
        let ssrCount = try make(label: "Illuminatorama.svgf.ssrCount",
                                format: .r16Float, w: internalW, h: internalH,
                                usage: [.shaderRead, .shaderWrite])
        let giCount = try make(label: "Illuminatorama.svgf.giCount",
                               format: .r16Float, w: internalW, h: internalH,
                               usage: [.shaderRead, .shaderWrite])
        let giVar = try make(label: "Illuminatorama.svgf.giVariance",
                             format: .r16Float, w: internalW, h: internalH,
                             usage: [.shaderRead, .shaderWrite])
        let giVarB = try make(label: "Illuminatorama.svgf.giVarianceB",
                              format: .r16Float, w: internalW, h: internalH,
                              usage: [.shaderRead, .shaderWrite])
        let giAtrousA = try make(label: "Illuminatorama.svgf.giAtrousA",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        let giAtrousB = try make(label: "Illuminatorama.svgf.giAtrousB",
                                 format: .rgba16Float, w: internalW, h: internalH,
                                 usage: [.shaderRead, .shaderWrite])
        // Issue #65 — screen-space SSS buffers (full-res, idle unless sssStrength > 0).
        let sssDiffuse = try make(label: "Illuminatorama.sss.diffuse",
                                  format: .rgba16Float, w: internalW, h: internalH,
                                  usage: [.shaderRead, .shaderWrite])
        let sssBlurTmp = try make(label: "Illuminatorama.sss.blurTmp",
                                  format: .rgba16Float, w: internalW, h: internalH,
                                  usage: [.shaderRead, .shaderWrite])
        let sssBlurOut = try make(label: "Illuminatorama.sss.blurOut",
                                  format: .rgba16Float, w: internalW, h: internalH,
                                  usage: [.shaderRead, .shaderWrite])
        // Bloom mip pyramid (S1.2). Level 0 is half the internal resolution — the
        // same size and format the old single-gaussian target was, which is why
        // the tonemap's binding and sampler are untouched by this change.
        let bloomLevels = Self.bloomLevelCount(halfW: halfW, halfH: halfH)
        var bloomDown: [MTLTexture] = []
        var bloomUp: [MTLTexture] = []
        for level in 0..<bloomLevels {
            let (lw, lh) = Self.bloomLevelSize(halfW: halfW, halfH: halfH, level: level)
            bloomDown.append(try make(label: "Illuminatorama.bloom.down\(level)",
                                      format: .rgba16Float, w: lw, h: lh,
                                      usage: [.shaderRead, .shaderWrite]))
            // The SMALLEST level gets no up-chain buffer: the up pass is seeded by
            // reading `bloomDown.last` directly, so allocating one would be a
            // texture nothing ever writes.
            if level < bloomLevels - 1 {
                bloomUp.append(try make(label: "Illuminatorama.bloom.up\(level)",
                                        format: .rgba16Float, w: lw, h: lh,
                                        usage: [.shaderRead, .shaderWrite]))
            }
        }
        // Film halation at HALF res — the halo's orange core is only a few output pixels
        // wide. Allocated unconditionally so the tonemap always has a texture to bind at
        // fragment slot 8; the passes that fill it are skipped when halationIntensity == 0.
        // `bright` holds one scalar (the red-weighted exposure), `blurH` four (one per scale).
        let hb = try make(label: "Illuminatorama.halation.bright",
                          format: .r16Float, w: halfW, h: halfH,
                          usage: [.shaderRead, .shaderWrite])
        let hbh = try make(label: "Illuminatorama.halation.blurH",
                           format: .rgba16Float, w: halfW, h: halfH,
                           usage: [.shaderRead, .shaderWrite])
        let hbv = try make(label: "Illuminatorama.halation.blurV",
                           format: .rgba16Float, w: halfW, h: halfH,
                           usage: [.shaderRead, .shaderWrite])
        // Final LDR output at OUTPUT resolution — the only texture sized to
        // what the SCNView shows. Tonemap kernel downsamples from internal
        // HDR + bloom into this each frame.
        let ldrDesc = MTLTextureDescriptor()
        ldrDesc.textureType = .type2D
        // sRGB so the tonemap can write linear + the GPU encodes on store; see
        // `makeLDRTexture`. sRGB targets disallow `.shaderWrite` (render-pass only).
        ldrDesc.pixelFormat = .bgra8Unorm_srgb
        ldrDesc.width = outputW
        ldrDesc.height = outputH
        ldrDesc.usage = [.shaderRead, .renderTarget]
        ldrDesc.storageMode = .private
        guard let ldr = device.makeTexture(descriptor: ldrDesc) else {
            throw IlluminatoramaError.bufferAllocationFailed("ldr")
        }
        ldr.label = "Illuminatorama.ldr"
        return Targets(
            albedoMet: am, normalRgh: nr, emission: em, depth: depth,
            hdr: hdr, ao: ao, hdrComposite: hdrComposite, rtDiffuse: rtDiffuse,
            velocity: velocity, layer: layer, historyA: historyA, historyB: historyB,
            taaResolved: taaResolved,
            bloomDown: bloomDown, bloomUp: bloomUp,
            halationBright: hb, halationBlurH: hbh, halationBlurV: hbv,
            ldr: ldr,
            aoFiltered: aoFiltered, aoHistoryA: aoHistA, aoHistoryB: aoHistB,
            ssrRaw: ssrRaw, ssrHistoryA: ssrHistA, ssrHistoryB: ssrHistB,
            rtGIHistoryA: rtGiHistA, rtGIHistoryB: rtGiHistB,
            aoSampleCount: aoCount, ssrSampleCount: ssrCount, giSampleCount: giCount,
            giVariance: giVar, giVarianceB: giVarB, giAtrousA: giAtrousA, giAtrousB: giAtrousB,
            sssDiffuse: sssDiffuse, sssBlurTmp: sssBlurTmp, sssBlurOut: sssBlurOut
        )
    }
}
