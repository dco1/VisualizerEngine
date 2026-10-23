import Metal
import simd

/// State of the accumulated cumulus in-view pass (`cumulusInViewEnabled`, DH-0578).
final class CumulusInViewState {
    /// Running sum per accumulator texel: rgb = Σ (marched − dome), a = sample count. Float32,
    /// at 1/`cumulusInViewScale` of the canvas.
    var accum: MTLTexture?
    /// What the accumulator was built for: canvas size, quality knobs and the sky (camera excluded).
    var cfgKey: UInt64?
    /// The camera it was built from (view + the projection's scale terms).
    var camKey: UInt64?
    /// Frames marched since either key last changed.
    var frame = 0
    /// Frames the camera and sky have held since either key last changed (keeps counting after the
    /// march stops; drives the reprojection crossfade).
    var stillFrames = 0
    var needsClear = true
    /// The camera `accum` is being built from (unjittered view-projection, eye).
    var anchorViewProj = matrix_identity_float4x4
    var anchorCam = SIMD3<Float>(repeating: 0)
    /// `cumulusInViewReprojection` — the last CONVERGED accumulator and the camera it was built from;
    /// reprojected into every frame while the camera moves and until the new one has converged.
    var history: MTLTexture?
    var historyViewProj = matrix_identity_float4x4
    var historyCam = SIMD3<Float>(repeating: 0)
    var historyValid = false

    func reset() { cfgKey = nil; camKey = nil; frame = 0; stillFrames = 0; historyValid = false }
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

/// Metal twin of `CumulusReprojectUniforms` (VolumetricSky.metal).
struct CumulusReprojectUniforms {
    var invViewProjection: simd_float4x4
    var historyViewProj: simd_float4x4
    var cameraWorldPos: SIMD4<Float>   // w = crossfade α (0 = history only, 1 = the new accumulator)
    var historyCam: SIMD4<Float>       // w = edge fade width (history uv)
    var parallax: SIMD4<Float>         // x = mid-slab Y, y/z = translation fade (m), w = far distance
    var cfg: SIMD4<UInt32>             // x = scale, y = history valid, zw = canvas size
}

extension IlluminatoramaRenderer {

    /// Samples per accumulator texel so far (0 until the first full interleave cycle lands).
    /// A host or gate reads it to know the sky has converged to the screen-resolution march.
    public var cumulusInViewSamplesPerPixel: Int {
        let n = cumulusInViewInterleaveClamped
        return cumulusInView.frame / (n * n)
    }

    var cumulusInViewInterleaveClamped: Int { cumulusInViewInterleave >= 4 ? 4 : (cumulusInViewInterleave >= 2 ? 2 : 1) }

    private func cumulusAccumTexture(_ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: w, height: h,
                                                         mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite, .renderTarget]
        d.storageMode = .private
        let t = device.makeTexture(descriptor: d)
        t?.label = label
        return t
    }

    /// The accumulated screen-resolution cumulus (DH-0578): march one interleave cell of the sky
    /// per frame while camera and sky hold still, then add the converged mean (march − dome) onto
    /// the dome-shaded sky pixels. Changing the sky resets it, and until a full cycle has landed
    /// the canvas shows the dome exactly as before — so an orbiting camera pays only the key check,
    /// and a converged one only the resolve.
    ///
    /// Moving the CAMERA: with `cumulusInViewReprojection` off it resets too (the sharp clouds pop
    /// back to the dome, then pop in again ~a cycle after the camera stops). On, the last converged
    /// accumulator is kept with the camera it was built from and REPROJECTED into each frame — the
    /// deck is ~a kilometre up, so a pixel's cloud is found in the old frame by projecting the point
    /// where its ray meets the middle of the slab (rotation exact, translation corrected by the
    /// slab's parallax). A moving camera keeps the sharp sky (no march: the cost is one resolve);
    /// once it stops, the new accumulator builds behind the reprojected one and crossfades in over
    /// `cumulusInViewFadeFrames`.
    func encodeCumulusInViewPass(_ cb: MTLCommandBuffer) {
        let st = cumulusInView
        guard cumulusInViewEnabled,
              let noise = cloudNoiseTexture,
              let skyU = cloudSkyUniforms,
              let dome = equirectSky,
              let resolve = engine.pipelineCache.pipelineState(name: "illumi_cumulus_inview_resolve", device: device),
              let hdr = pfx.hdrCompositeTexture
        else { st.reset(); return }
        let reproj = cumulusInViewReprojection
        let W = hdr.width, H = hdr.height
        let sc = cumulusInViewScale >= 2 ? 2 : 1
        let AW = (W + sc - 1) / sc, AH = (H + sc - 1) / sc
        if st.accum?.width != AW || st.accum?.height != AH {
            st.accum = cumulusAccumTexture(AW, AH, "Illuminatorama.cumulusInView.accum")
            st.history = nil
            st.reset()
        }
        if reproj, st.history == nil {
            st.history = cumulusAccumTexture(AW, AH, "Illuminatorama.cumulusInView.history")
            st.historyValid = false
        }
        guard let _ = st.accum else { return }

        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        let n = cumulusInViewInterleaveClamped
        let cells = n * n
        let detail = max(0, min(2, cumulusInViewDetail))
        let steps = max(16, cumulusInViewSteps), lightSteps = max(2, cumulusInViewLightSteps)
        let maxSamples = max(1, cumulusInViewMaxSamples)
        let cubic: UInt32 = fu.nightSkyExtra.x > 0.5 ? 1 : 0

        // ── Keys (FNV-1a). Camera: the view + the projection's scale terms — NOT its jitter column.
        // Config: canvas, quality and the host-owned bytes of the sky uniforms MINUS the camera
        // position (the first float4 — the camera key carries it; the GPU-written tail is re-derived
        // from the host bytes, so it is excluded).
        func fnv(_ body: (_ mix: (UInt32) -> Void) -> Void) -> UInt64 {
            var h: UInt64 = 0xcbf29ce484222325
            body { v in h = (h ^ UInt64(v)) &* 0x100000001b3 }
            return h
        }
        let camKey = fnv { mix in
            for c in 0..<4 { for r in 0..<4 { mix(fu.view[c][r].bitPattern) } }
            mix(fu.projection[0][0].bitPattern); mix(fu.projection[1][1].bitPattern)
        }
        let cfgKey = fnv { mix in
            mix(UInt32(W)); mix(UInt32(H)); mix(UInt32(sc)); mix(UInt32(n)); mix(UInt32(steps))
            mix(UInt32(lightSteps)); mix(cubic); mix(detail.bitPattern); mix(UInt32(maxSamples))
            let words = SkyUniforms.hostPrefixLength / 4
            let p = skyU.contents().bindMemory(to: UInt32.self, capacity: words)
            for i in 4..<words { mix(p[i]) }
        }
        let viewProj = camera.projectionMatrix * fu.view
        let cam = fu.cameraWorldPos
        func restart() {
            st.camKey = camKey; st.frame = 0; st.stillFrames = 0; st.needsClear = true
            st.anchorViewProj = viewProj; st.anchorCam = cam
        }
        // A key that changed THIS frame means the camera or sky is moving: march nothing (an orbit
        // would otherwise pay one interleave cell every frame for an accumulator it throws away).
        if st.cfgKey != cfgKey {
            st.cfgKey = cfgKey; st.historyValid = false; restart(); return
        }
        if st.camKey != camKey {
            if reproj, st.frame >= cells, let a = st.accum, let b = st.history {
                // The accumulator converged (at least one full cycle): it becomes the history.
                st.accum = b; st.history = a
                st.historyViewProj = st.anchorViewProj; st.historyCam = st.anchorCam
                st.historyValid = true
            }
            restart()
            if reproj, st.historyValid { encodeCumulusReprojectResolve(cb, hdr: hdr, skyU: skyU, sc: sc, alpha: 0) }
            return
        }
        guard let accum = st.accum else { return }

        if st.needsClear {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = accum
            rp.colorAttachments[0].loadAction = .clear
            rp.colorAttachments[0].storeAction = .store
            rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            cb.makeRenderCommandEncoder(descriptor: rp)?.endEncoding()
            st.needsClear = false
        }
        st.stillFrames += 1

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

        if reproj {
            // Crossfade from the reprojected history to the new accumulator once it has a full cycle.
            let fade = max(0, cumulusInViewFadeFrames)
            let since = st.stillFrames - cells   // ≥ 0 once the first full cycle has landed
            let alpha: Float = st.frame < cells ? 0 : (fade == 0 ? 1 : min(1, Float(since + 1) / Float(fade)))
            if alpha < 1, st.historyValid || alpha > 0 {
                encodeCumulusReprojectResolve(cb, hdr: hdr, skyU: skyU, sc: sc, alpha: alpha)
                return
            }
            if alpha < 1 { return }
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

    /// The reprojecting resolve: the history (warped from its own camera into this one, faded at its
    /// frame edges and with distance from where it was built) blended toward the new accumulator by α.
    private func encodeCumulusReprojectResolve(_ cb: MTLCommandBuffer, hdr: MTLTexture, skyU: MTLBuffer,
                                               sc: Int, alpha: Float) {
        let st = cumulusInView
        guard let accum = st.accum, let history = st.history,
              let pipe = engine.pipelineCache.pipelineState(name: "illumi_cumulus_inview_resolve_reproject",
                                                             device: device),
              let enc = timedComputeEncoder(cb, "cloudCumulus.resolve") else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        let slab = skyU.contents().load(fromByteOffset: MemoryLayout<SkyUniforms>.offset(of: \SkyUniforms.cloudSlab)!,
                                        as: SIMD4<Float>.self)
        let base = max(slab.x, 50)
        var u = CumulusReprojectUniforms(
            invViewProjection: fu.invViewProjection,
            historyViewProj: st.historyViewProj,
            cameraWorldPos: SIMD4<Float>(fu.cameraWorldPos, alpha),
            historyCam: SIMD4<Float>(st.historyCam, 0.06),
            // Trust the history fully within 3 % of the deck's height of where it was built and not at
            // all past 12 % (36 / 144 m at a 1.2 km base): beyond that the single mid-slab depth's
            // parallax error grows past a few pixels near the zenith.
            parallax: SIMD4<Float>(0.5 * (slab.x + slab.y), 0.03 * base, 0.12 * base, 30_000),
            cfg: SIMD4<UInt32>(UInt32(sc), st.historyValid ? 1 : 0, UInt32(hdr.width), UInt32(hdr.height)))
        enc.label = "Illuminatorama.cumulusInView.resolveReproject"
        enc.setComputePipelineState(pipe)
        enc.setTexture(hdr, index: 0)
        enc.setTexture(accum, index: 1)
        enc.setTexture(gbs.depthTexture, index: 2)
        enc.setTexture(history, index: 3)
        enc.setBytes(&u, length: MemoryLayout<CumulusReprojectUniforms>.stride, index: 0)
        dispatch(enc, pipeline: pipe, width: hdr.width, height: hdr.height)
        enc.endEncoding()
    }
}
