import Metal
import simd

/// State of the accumulated cumulus in-view pass (`cumulusInViewEnabled`, DH-0578).
final class CumulusInViewState {
    /// Running sum per accumulator texel: rgb = Σ (marched − dome), a = sample count. Float32,
    /// at 1/`cumulusInViewScale` of the canvas.
    var accum: MTLTexture?
    /// What the accumulator was built for (camera, canvas size, sky uniforms, quality knobs).
    var key: UInt64?
    /// Frames marched since `key` last changed.
    var frame = 0
    var needsClear = true
}

/// Metal twin of `CumulusInViewUniforms` (VolumetricSky.metal).
struct CumulusInViewUniforms {
    var invViewProjection: simd_float4x4
    var cameraWorldPos: SIMD4<Float>
    var grid: SIMD4<UInt32>
    var canvas: SIMD4<UInt32>
    var quality: SIMD4<Float>
    var samples: SIMD4<UInt32>
}

extension IlluminatoramaRenderer {

    /// Samples per accumulator texel so far (0 until the first full interleave cycle lands).
    /// A host or gate reads it to know the sky has converged to the screen-resolution march.
    public var cumulusInViewSamplesPerPixel: Int {
        let n = cumulusInViewInterleaveClamped
        return cumulusInView.frame / (n * n)
    }

    var cumulusInViewInterleaveClamped: Int { cumulusInViewInterleave >= 4 ? 4 : (cumulusInViewInterleave >= 2 ? 2 : 1) }

    /// The accumulated screen-resolution cumulus (DH-0578): march one interleave cell of the sky
    /// per frame while camera and sky hold still, then add the converged mean (march − dome) onto
    /// the dome-shaded sky pixels. Moving the camera or changing the sky resets it, and until a
    /// full cycle has landed the canvas shows the dome exactly as before — so an orbiting camera
    /// pays only the key check, and a converged one only the resolve.
    func encodeCumulusInViewPass(_ cb: MTLCommandBuffer) {
        let st = cumulusInView
        guard cumulusInViewEnabled,
              let noise = cloudNoiseTexture,
              let skyU = cloudSkyUniforms,
              let dome = equirectSky,
              let resolve = engine.pipelineCache.pipelineState(name: "illumi_cumulus_inview_resolve", device: device),
              let hdr = pfx.hdrCompositeTexture
        else { st.key = nil; st.frame = 0; return }
        let W = hdr.width, H = hdr.height
        let sc = cumulusInViewScale >= 2 ? 2 : 1
        let AW = (W + sc - 1) / sc, AH = (H + sc - 1) / sc
        if st.accum?.width != AW || st.accum?.height != AH {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: AW, height: AH,
                                                             mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite, .renderTarget]
            d.storageMode = .private
            st.accum = device.makeTexture(descriptor: d)
            st.accum?.label = "Illuminatorama.cumulusInView.accum"
            st.key = nil
        }
        guard let accum = st.accum else { return }

        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        let n = cumulusInViewInterleaveClamped
        let cells = n * n
        let detail = max(0, min(2, cumulusInViewDetail))
        let steps = max(16, cumulusInViewSteps), lightSteps = max(2, cumulusInViewLightSteps)
        let maxSamples = max(1, cumulusInViewMaxSamples)
        let cubic: UInt32 = fu.nightSkyExtra.x > 0.5 ? 1 : 0

        // ── Key: camera (view + the projection's scale terms — NOT its jitter column), canvas,
        // quality, and the host-owned bytes of the sky uniforms (FNV-1a; the GPU-written tail is
        // re-derived from those bytes, so it is excluded).
        var h: UInt64 = 0xcbf29ce484222325
        func mix(_ v: UInt32) { h = (h ^ UInt64(v)) &* 0x100000001b3 }
        func mixF(_ f: Float) { mix(f.bitPattern) }
        for c in 0..<4 { for r in 0..<4 { mixF(fu.view[c][r]) } }
        mixF(fu.projection[0][0]); mixF(fu.projection[1][1])
        mix(UInt32(W)); mix(UInt32(H)); mix(UInt32(sc)); mix(UInt32(n)); mix(UInt32(steps))
        mix(UInt32(lightSteps)); mix(cubic); mixF(detail); mix(UInt32(maxSamples))
        let words = SkyUniforms.hostPrefixLength / 4
        let p = skyU.contents().bindMemory(to: UInt32.self, capacity: words)
        for i in 0..<words { mix(p[i]) }
        // A key that changed THIS frame means the camera or sky is moving: march nothing (an orbit
        // would otherwise pay one interleave cell every frame for an accumulator it throws away).
        if st.key != h { st.key = h; st.frame = 0; st.needsClear = true; return }

        if st.needsClear {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = accum
            rp.colorAttachments[0].loadAction = .clear
            rp.colorAttachments[0].storeAction = .store
            rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            cb.makeRenderCommandEncoder(descriptor: rp)?.endEncoding()
            st.needsClear = false
        }

        if st.frame < cells * maxSamples {
            let ms = VolumetricCloudRenderer.multipleScatteringRequested(in: skyU)
            let aerosol = VolumetricCloudRenderer.aerosolRequested(in: skyU)
            let name = "illumi_cumulus_inview" + (ms ? "_ms" : "") + (aerosol ? "_aerosol" : "")
            if let march = engine.pipelineCache.pipelineState(name: name, device: device),
               let enc = timedComputeEncoder(cb, "cloudCumulus.march") {
                // Bayer order, so each frame's cell lands as far as possible from the last.
                let order4 = [0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12]
                let order2 = [0, 3, 1, 2]
                let k = st.frame % cells
                let cell = n == 4 ? order4[k] : (n == 2 ? order2[k] : 0)
                let pixelAngle = (2 / max(fu.projection[1][1], 1e-6)) / Float(H)
                var u = CumulusInViewUniforms(
                    invViewProjection: fu.invViewProjection,
                    cameraWorldPos: SIMD4<Float>(fu.cameraWorldPos, pixelAngle),
                    grid: SIMD4<UInt32>(UInt32(n), UInt32(cell % n), UInt32(cell / n), UInt32(st.frame / cells)),
                    canvas: SIMD4<UInt32>(UInt32(sc), cubic, UInt32(W), UInt32(H)),
                    quality: SIMD4<Float>(detail, Float(steps), Float(lightSteps), ms ? 1 : 0),
                    samples: SIMD4<UInt32>(UInt32(maxSamples), 0, 0, 0))
                enc.label = "Illuminatorama.cumulusInView.march"
                enc.setComputePipelineState(march)
                enc.setTexture(accum, index: 0)
                enc.setTexture(noise, index: 1)
                enc.setTexture(gbs.depthTexture, index: 2)
                enc.setTexture(dome, index: 3)
                enc.setBuffer(skyU, offset: 0, index: 0)
                enc.setBytes(&u, length: MemoryLayout<CumulusInViewUniforms>.stride, index: 1)
                enc.setBuffer(skyU, offset: ms ? VolumetricCloudRenderer.atmosphereLUTOffset : 0, index: 2)
                dispatch(enc, pipeline: march, width: (AW + n - 1) / n, height: (AH + n - 1) / n)
                enc.endEncoding()
                st.frame += 1
            }
        }

        // Add the delta once every texel has a sample.
        guard st.frame >= cells, let enc = timedComputeEncoder(cb, "cloudCumulus.resolve") else { return }
        var cfg = SIMD4<UInt32>(UInt32(sc), 0, 0, 0)
        enc.label = "Illuminatorama.cumulusInView.resolve"
        enc.setComputePipelineState(resolve)
        enc.setTexture(hdr, index: 0)
        enc.setTexture(accum, index: 1)
        enc.setTexture(gbs.depthTexture, index: 2)
        enc.setBytes(&cfg, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
        dispatch(enc, pipeline: resolve, width: W, height: H)
        enc.endEncoding()
    }
}
