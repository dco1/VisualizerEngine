import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - VolumetricState

/// Volumetric shafts, spot-beam scattering and in-view clouds.
/// Owned by the renderer as `vols`; every member keeps the name it had on the renderer.
@MainActor
final class VolumetricState {
    var volRTProgressiveIndex: UInt32 = 0

    var volPipeline: MTLComputePipelineState?
    var volUniformBuffer: MTLBuffer!
    var volFrameSeed: UInt32 = 0

    var volSpotPipeline: MTLComputePipelineState?
    var volSpotUniformBuffer: MTLBuffer!

    var cloudInViewPipeline: MTLComputePipelineState?

    /// Last frame's value, for motion vectors.
    var lastFieldSinkProgress: Float = 0

    var cloudLowResTexture: MTLTexture?
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// Mirror of the Metal `VolRTUniforms` (IlluminatoramaRTInstanced.metal).
    struct VolRTUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var density: Float
        var sunDir: SIMD3<Float>; var anisotropy: Float
        var sunColor: SIMD3<Float>; var skyIntensity: Float
        var width: UInt32; var height: UInt32; var steps: UInt32; var progressiveIndex: UInt32
        var maxDist: Float; var roofReach: Float; var areaLightCount: UInt32; var transportMask: UInt32
        var sunScatter: Float; var skyScatter: Float; var scotopicDesaturation: Float; var rayTMin: Float
        var dustDensity: Float; var dustAnisotropy: Float; var dustAdmittanceRef: Float; var admittanceGrid: UInt32
        var pixelStride: UInt32 = 1
    }

    /// Drive the analytic (screen-resolution) PHYSICAL night sky from the same
    /// `VolumetricCloudRenderer.Params` the host renders its sky dome with — moon, stars,
    /// gains and celestial frame in one call, so the dome (its moonlit glow and clouds) and
    /// the per-pixel celestials cannot disagree. Pair it with `params.nightSkyModel =
    /// .physical` and `params.celestialsInDome = false`: the dome then writes its cloud
    /// transmittance to alpha and the stars / moon go behind the clouds. Stars and the Milky
    /// Way are skipped while the sun is up (they would wash out anyway); the moon is always
    /// drawn — a daytime moon is a pale disk.
    public func setPhysicalNightSky(from p: VolumetricCloudRenderer.Params) {
        let sunTravel = p.sunDir == .zero ? SIMD3<Float>(0, -1, 0) : simd_normalize(p.sunDir)
        let moon = p.moonDir == .zero ? SIMD3<Float>(0.4, 0.6, -0.7) : simd_normalize(p.moonDir)
        let night = VolumetricCloudRenderer.nightBlend(sunDir: sunTravel) > 0
        nightSkyModel = .physical
        nightSkyStarBrightness = night ? max(0, p.starBrightness) : 0
        nightSkyMilkyWay = night ? max(0, p.milkyWay) : 0
        nightSkyMoonIntensity = max(0, p.moonIntensity)
        nightSkyMoonDirection = moon
        nightSkySunDirection = p.moonPhaseOverride.map {
            NightSkyEphemeris.effectiveSun(moonDir: moon, trueToSun: -sunTravel, illuminatedFraction: $0)
        } ?? -sunTravel
        nightSkyMoonAngularRadius = max(0, p.moonAngularRadius)   // 0 = the real 0.26°
        nightSkyMoonHalo = max(0, p.moonHalo)
        nightSkyEarthshine = max(0, p.earthshine)
        nightSkyTwinkle = max(0, p.starTwinkle)
        nightSkyRadiance = max(0, p.nightRadiance)
        nightSkyCelestialOrientation = p.celestialOrientation
        nightSkyStarLimitingMagnitude = max(0, p.starLimitingMagnitude)
    }

    /// The physical-night-sky uniform clusters, packed ONCE for the frame and the glass pass
    /// (so a window and the sky beside it cannot disagree). `.legacy` ⇒ all zero ⇒ the
    /// shader's legacy branch, byte-identical.
    func physicalNightSkyClusters() -> (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) {
        guard nightSkyModel == .physical else { return (.zero, .zero, .zero) }
        let q = nightSkyCelestialOrientation
        let qn = simd_length(q) > 1e-6 ? q / simd_length(q) : SIMD4<Float>(0, 0, 0, 1)
        return (SIMD4(1, max(0, nightSkyMoonHalo), max(0, nightSkyEarthshine), max(0, nightSkyMilkyWay)),
                SIMD4(max(0, nightSkyTwinkle), max(0, nightSkyRadiance), time, max(0, nightSkyStarLimitingMagnitude)),
                qn)
    }

    // ── Volumetric shaft state ───────────────────────────────────────
    struct VolUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var fogDensity: Float
        var sunDir: SIMD3<Float>; var scatterStrength: Float
        var sunColor: SIMD3<Float>; var anisotropy: Float
        var windowX: Float; var winY0: Float; var winY1: Float; var winZ0: Float
        var winZ1: Float; var maxDist: Float; var steps: UInt32; var frameSeed: UInt32
        var width: UInt32; var height: UInt32; var feather: Float; var isOutdoor: UInt32 = 0
    }

    // ── Spot-beam scattering state ───────────────────────────────────
    // Mirror of the Metal `VolSpotUniforms` in IlluminatoramaVolumetric.metal.
    struct VolSpotUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var density: Float
        var anisotropy: Float; var strength: Float; var spotCount: UInt32; var steps: UInt32
        var maxDist: Float; var frameSeed: UInt32; var width: UInt32; var height: UInt32
        // Emissive column — mirrors the Metal `colTop … colSteps` block (all zero ⇒ off).
        var colTop: SIMD4<Float> = .zero
        var colAxis: SIMD4<Float> = .zero
        var colShape: SIMD4<Float> = .zero
        var colSolid: SIMD4<Float> = .zero
        var colBand: SIMD4<Float> = .zero
        var colSteps: UInt32 = 0
        /// 1 ⇒ each beam sample is tested against its spot's shadow slice (`volSpotBeamShadows`).
        var beamShadows: UInt32 = 0
        /// Depth bias for the beam shadow compare — the host's `spotShadowBias`, so beam and
        /// surface shadow edges agree.
        var beamShadowBias: Float = 0
        var _colPad2: UInt32 = 0
    }

    // ── In-view (perspective) cloud pass — OPT-IN ────────────────────
    // Mirror of the Metal `CloudInViewUniforms`. The cloud density/lighting/
    // burst params ride in the cloud renderer's own SkyUniforms buffer
    // (`cloudSkyUniforms`); this only adds the camera inverse-VP + eye so the
    // kernel can reconstruct world rays per pixel.
    struct CloudInViewUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD4<Float>
        /// x = frame time (s) — per-frame sky animation (lava lamp) in the in-view pass.
        /// y = per-frame lava-lamp weight override (≥ 0), or −1 = use SkyUniforms.lavaA.x.
        /// z = cirrus clock override (≥ 0), or −1 = x.
        /// w = cirrus threads weight override (0…1), or −1 = SkyUniforms.cirrusC.x.
        var extra: SIMD4<Float> = .zero
        /// x = the renderer's frame clock (s) — stellar scintillation in the physical night
        /// sky (per frame, so a twinkle never steps at the host's sky-update rate).
        /// y = 1 when the multiple-scattering kernel runs with the atmosphere LUTs at buffer(3).
        var night: SIMD4<Float> = .zero
    }

    /// Single-scatter volumetric light shaft, added into the HDR composite
    /// after the RT lighting (so the beam glows in the air in front of lit
    /// surfaces) and before particles / TAA / bloom. No-op unless enabled.
    func encodeVolumetricPass(_ cb: MTLCommandBuffer) {
        guard volumetricEnabled, let pipeline = vols.volPipeline else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        vols.volFrameSeed &+= 1
        var u = VolUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos, fogDensity: max(0, volFogDensity),
            sunDir: simd_normalize(rtSunDirection), scatterStrength: max(0, volScatterStrength),
            sunColor: rtSunColor, anisotropy: volAnisotropy,
            windowX: volWindowX, winY0: volWindowRect.x, winY1: volWindowRect.y,
            winZ0: volWindowRect.z, winZ1: volWindowRect.w, maxDist: volMaxDistance,
            steps: UInt32(max(1, min(96, volSteps))), frameSeed: vols.volFrameSeed,
            width: UInt32(width), height: UInt32(height), feather: volFeather,
            isOutdoor: volOutdoorMode ? 1 : 0)
        memcpy(vols.volUniformBuffer.contents(), &u, MemoryLayout<VolUniforms>.stride)
        guard let enc = timedComputeEncoder(cb, "volumetric") else { return }
        enc.label = "Illuminatorama.volumetric"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(gbs.depthTexture, index: 0)
        enc.setTexture(pfx.hdrCompositeTexture, index: 1)
        enc.setBuffer(vols.volUniformBuffer, offset: 0, index: 0)
        enc.setBuffer(cloudShadowBinding, offset: 0, index: 1)   // DH-1492 — cloud sun transmittance
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// DH-0990 — the traced volumetric light (see `volumetricRTEnabled`). After the RT lighting and
    /// its denoise chain have composited, so the haze attenuates the finished surface radiance it
    /// sits in front of; before glass, TAA and bloom.
    func encodeVolumetricRTPass(_ cb: MTLCommandBuffer) {
        // A TLAS holding CURVE instances needs the curve-aware intersector contract this kernel does
        // not compile; such a scene (no Daydream document today) skips the haze rather than trace it
        // under the wrong geometry-type assumption.
        guard volumetricRTEnabled, volRTDensity > 0, rts.rtTLASActive, rts.rtCurveInstanceCount == 0,
              let tlas = rts.rtTLAS, let pipeline = volRTPipeline else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        var u = VolRTUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos, density: volRTDensity,
            sunDir: simd_normalize(rtSunDirection), anisotropy: max(-0.95, min(0.95, volRTAnisotropy)),
            sunColor: rtSunColor, skyIntensity: max(0, iblIntensity),
            width: UInt32(width), height: UInt32(height),
            steps: UInt32(max(1, min(64, volRTSteps))), progressiveIndex: vols.volRTProgressiveIndex,
            maxDist: max(0.1, volRTMaxDistance), roofReach: max(0.1, volRTRoofReach),
            areaLightCount: UInt32(areaLights.count), transportMask: Self.rtTransportRayMask,
            sunScatter: max(0, volRTSunScatter), skyScatter: max(0, volRTSkyScatter),
            scotopicDesaturation: max(0, scotopicDesaturation), rayTMin: 0.004,
            dustDensity: max(0, volRTDustDensity), dustAnisotropy: max(-0.95, min(0.95, volRTDustAnisotropy)),
            dustAdmittanceRef: max(1e-4, volRTDustAdmittanceRef),
            admittanceGrid: UInt32(max(1, min(512, volRTAdmittanceGrid))),
            pixelStride: UInt32(max(1, min(16, volRTPixelStride))))
        vols.volRTProgressiveIndex &+= 1
        guard let counts = volRTAdmittanceBuffer else { return }
        // DH-0999 — measure the direct sun entering the view's air BEFORE the march reads it. The
        // counters are cleared every haze frame; with the dust off the pass is skipped and the
        // zeroed counters read as "no dust".
        if let blit = cb.makeBlitCommandEncoder() {
            blit.label = "Illuminatorama.volumetricRT.admittanceClear"
            blit.fill(buffer: counts, range: 0..<counts.length, value: 0)
            blit.endEncoding()
        }
        if u.dustDensity > 0, let admPipeline = volRTAdmittancePipeline,
           let enc = timedComputeEncoder(cb, "volumetricRTAdmittance") {
            enc.label = "Illuminatorama.volumetricRT.admittance"
            enc.setComputePipelineState(admPipeline)
            enc.setTexture(gbs.depthTexture, index: 0)
            enc.setAccelerationStructure(tlas, bufferIndex: 0)
            enc.setBytes(&u, length: MemoryLayout<VolRTUniforms>.stride, index: 1)
            enc.setBuffer(counts, offset: 0, index: 2)
            enc.setBuffer(cloudShadowBinding, offset: 0, index: 3)   // DH-1492
            for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
            for blas in rts.rtCurveBLASList { enc.useResource(blas, usage: .read) }
            for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
            let gx = Int(u.admittanceGrid)
            let gy = max(1, Int(Float(gx) * Float(height) / Float(max(width, 1))))
            dispatch(enc, pipeline: admPipeline, width: gx, height: gy, maxThreadsOverride: rtThreadgroupMax)
            enc.endEncoding()
        }
        guard let enc = timedComputeEncoder(cb, "volumetricRT") else { return }
        enc.label = "Illuminatorama.volumetricRT"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(gbs.depthTexture, index: 0)
        enc.setTexture(pfx.hdrCompositeTexture, index: 1)
        enc.setTexture(equirectSky ?? ibls.dummySkyTexture, index: 2)
        enc.setAccelerationStructure(tlas, bufferIndex: 0)
        enc.setBytes(&u, length: MemoryLayout<VolRTUniforms>.stride, index: 1)
        enc.setBuffer(areaLightBuffer, offset: 0, index: 2)
        enc.setBuffer(counts, offset: 0, index: 3)
        enc.setBuffer(cloudShadowBinding, offset: 0, index: 4)   // DH-1492 — cloud sun transmittance
        for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
        for blas in rts.rtCurveBLASList { enc.useResource(blas, usage: .read) }
        for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
        dispatch(enc, pipeline: pipeline, width: width, height: height, maxThreadsOverride: rtThreadgroupMax)
        enc.endEncoding()
    }

    /// Spot-light beam scattering — every entry in `spotLights` marches as a
    /// haze-lit cone. Runs after the sun volumetric pass so both add into the
    /// same HDR composite; independent of `volumetricEnabled`.
    func encodeSpotBeamPass(_ cb: MTLCommandBuffer) {
        let spotsOn = volSpotBeamsEnabled && !spotLights.isEmpty
        let column = volEmissiveColumn.flatMap { $0.front > 0 && $0.length > 0 ? $0 : nil }
        guard spotsOn || column != nil,
              let pipeline = vols.volSpotPipeline else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        vols.volFrameSeed &+= 1
        var u = VolSpotUniforms(
            invViewProjection: fu.invViewProjection,
            cameraWorldPos: fu.cameraWorldPos, density: max(0, volSpotBeamDensity),
            anisotropy: volSpotBeamAnisotropy, strength: max(0, volSpotBeamStrength),
            spotCount: spotsOn ? UInt32(min(spotLights.count, max(0, volSpotBeamSpotLimit))) : 0,
            steps: UInt32(max(4, min(48, volSpotBeamSteps))),
            maxDist: max(volSpotBeamMaxDistance, column != nil ? 400 : 0),
            frameSeed: vols.volFrameSeed,
            width: UInt32(width), height: UInt32(height))
        u.beamShadows = volSpotBeamShadows ? 1 : 0
        u.beamShadowBias = spotShadowBias
        if let c = column {
            let axis = simd_length(c.axis) > 1e-6 ? simd_normalize(c.axis) : SIMD3<Float>(0, -1, 0)
            u.colTop = SIMD4(c.top, 1)
            u.colAxis = SIMD4(axis, c.length)
            u.colShape = SIMD4(max(0, c.topRadius), max(0, c.bottomRadius), c.front, c.edgeFeather)
            u.colSolid = SIMD4(simd_max(c.solidRadiance, .zero), min(1, max(0, c.rainbowMix)))
            u.colBand = SIMD4(max(0, c.rainbowRadiance), c.rainbowCycles, c.rainbowScroll,
                              Float(max(0, c.rainbowSteps)))
            u.colSteps = UInt32(max(4, min(64, c.steps)))
        }
        guard let enc = timedComputeEncoder(cb, "volumetricSpots") else { return }
        enc.label = "Illuminatorama.volumetricSpots"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(gbs.depthTexture, index: 0)
        enc.setTexture(pfx.hdrCompositeTexture, index: 1)
        enc.setTexture(shd.spotShadowAtlas, index: 2)
        // Inline bytes, not the shared `volSpotUniformBuffer`: with 2 frames in flight a
        // memcpy into one buffer overwrote the params the previous frame's kernel was still
        // reading (the per-frame emissive column — its front, scroll, position — made that tear).
        enc.setBytes(&u, length: MemoryLayout<VolSpotUniforms>.stride, index: 0)
        enc.setBuffer(spotLightBuffer, offset: 0, index: 1)
        enc.setBuffer(shd.spotProfileBuffer, offset: 0, index: 2)   // DH-0540 — the beam carries the profile too
        dispatch(enc, pipeline: pipeline, width: width, height: height)
        enc.endEncoding()
    }

    /// In-view (perspective) cloud composite — issue #61. Raymarches the scene's
    /// cloud volume along per-pixel camera rays (crisp at output res, vs the
    /// equirect dome's upscaled-crop blur) and writes the result into
    /// `hdrCompositeTexture`, before bloom/tonemap and before the additive
    /// firework particles (which draw in `encodePostResolveFX`). Opt-in:
    /// no-ops unless the host scene enabled it and wired the reuse handles.
    /// v1 REPLACES sky pixels (the only opt-in scene has no opaque geometry); a
    /// depth-clipped composite-over-geometry generalisation is future work.
    func encodeCloudPass(_ cb: MTLCommandBuffer) {
        encodeCumulusInViewPass(cb)
        guard inViewCloudsEnabled,
              let pipeline = vols.cloudInViewPipeline,
              let noise = cloudNoiseTexture,
              let skyU = cloudSkyUniforms,
              let lights = cloudBurstLights else { return }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        // Downsample factor (1 = the original full-res pass, byte-identical).
        let factor = inViewCloudsResolutionScale < 0.99
            ? max(2, min(4, Int((1 / max(inViewCloudsResolutionScale, 0.25)).rounded()))) : 1
        var lowRes: MTLTexture? = nil
        if factor > 1 {
            let lw = (width + factor - 1) / factor, lh = (height + factor - 1) / factor
            if vols.cloudLowResTexture?.width != lw || vols.cloudLowResTexture?.height != lh {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: lw, height: lh,
                                                                 mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                vols.cloudLowResTexture = device.makeTexture(descriptor: d)
                vols.cloudLowResTexture?.label = "Illuminatorama.cloudInView.lowRes"
            }
            lowRes = cloudUpsamplePipeline == nil ? nil : vols.cloudLowResTexture
        }
        let f = lowRes == nil ? 1 : factor
        // Multiple scattering (`VolumetricCloudRenderer.Params.atmosphereMultipleScattering`):
        // its LUTs live after the uniforms in the cloud renderer's uniforms buffer, so this pass
        // draws the same sky as the dome and IBL with no extra wiring — `illumi_cloud_inview_ms`
        // with the LUT region at buffer(3) (night.y = bound). Off: the unchanged kernel.
        // An aerosol other than the built-in one (`Params.atmosphereAerosol`) rides in the same
        // uniforms: the `_aerosol` twins read it as the Mie medium (the unchanged kernels keep the
        // built-in constants — bit-identical for every host that never sets one).
        let aerosol = VolumetricCloudRenderer.aerosolRequested(in: skyU)
        let msPipeline = VolumetricCloudRenderer.multipleScatteringRequested(in: skyU)
            ? (aerosol ? cloudInViewPipelineMSAerosol : cloudInViewPipelineMS) : nil
        let marchPipeline = msPipeline ?? (aerosol ? (cloudInViewPipelineAerosol ?? pipeline) : pipeline)
        // (threadgroups are sized from the bound kernel)
        var u = CloudInViewUniforms(invViewProjection: fu.invViewProjection,
                                    cameraWorldPos: SIMD4<Float>(fu.cameraWorldPos, f > 1 ? Float(f) : 0),
                                    extra: SIMD4<Float>(inViewSkyTime ?? time, inViewLavaFade.map { min(max($0, 0), 1) } ?? -1,
                                                         inViewCirrusTime.map { max($0, 0) } ?? -1,
                                                         inViewCirrusThreads.map { min(max($0, 0), 1) } ?? -1),
                                    night: SIMD4<Float>(time, msPipeline != nil ? 1 : 0, 0, 0))
        // setBytes, not a shared MTLBuffer: with two frames in flight a single buffer was overwritten by frame N+1 while frame N's kernels could still read it (the
        // camera matrix the rays — and the upsample's stars — are rebuilt from).
        let uStride = MemoryLayout<CloudInViewUniforms>.stride
        guard let enc = timedComputeEncoder(cb, "cloudInView") else { return }
        enc.label = "Illuminatorama.cloudInView"
        enc.setComputePipelineState(marchPipeline)
        enc.setTexture(lowRes ?? pfx.hdrCompositeTexture, index: 0)
        enc.setTexture(noise, index: 1)
        // v2 depth clip — the kernel writes only where this says there is no
        // geometry, so the deck can no longer paint over the building.
        enc.setTexture(gbs.depthTexture, index: 2)
        enc.setBuffer(skyU, offset: 0, index: 0)
        enc.setBytes(&u, length: uStride, index: 1)
        enc.setBuffer(lights, offset: 0, index: 2)
        if msPipeline != nil { enc.setBuffer(skyU, offset: VolumetricCloudRenderer.atmosphereLUTOffset, index: 3) }
        if let lowRes {
            dispatch(enc, pipeline: marchPipeline, width: lowRes.width, height: lowRes.height)
            enc.endEncoding()
            guard let up = cloudUpsamplePipeline,
                  let enc2 = timedComputeEncoder(cb, "cloudInView.upsample") else { return }
            enc2.label = "Illuminatorama.cloudInView.upsample"
            enc2.setComputePipelineState(up)
            enc2.setTexture(pfx.hdrCompositeTexture, index: 0)
            enc2.setTexture(lowRes, index: 1)
            enc2.setTexture(gbs.depthTexture, index: 2)
            enc2.setBytes(&u, length: uStride, index: 0)
            enc2.setBuffer(skyU, offset: 0, index: 1)
            dispatch(enc2, pipeline: up, width: width, height: height)
            enc2.endEncoding()
        } else {
            dispatch(enc, pipeline: marchPipeline, width: width, height: height)
            enc.endEncoding()
        }
    }
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// Volumetric shaft, spot-beam scattering and in-view cloud pipelines (+ their uniform buffers).
    @MainActor static func buildVolumetricPipelines(into s: VolumetricState, device: MTLDevice, library: MTLLibrary) throws {
        // ── Volumetric shaft pipeline ────────────────────────────────
        if let volFn = library.makeFunction(name: "illumi_volumetric") {
            s.volPipeline = try? device.makeComputePipelineState(function: volFn) // gpu-ok: one-time init, optional volumetric pipeline
        } else {
            s.volPipeline = nil
        }
        guard let volUB = device.makeBuffer(length: MemoryLayout<VolUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("volUniforms")
        }
        volUB.label = "Illuminatorama.vol.uniforms"
        s.volUniformBuffer = volUB

        // ── Spot-beam scattering pipeline ────────────────────────────
        if let volSpotFn = library.makeFunction(name: "illumi_volumetric_spots") {
            s.volSpotPipeline = try? device.makeComputePipelineState(function: volSpotFn) // gpu-ok: one-time init, optional spot-beam pipeline
        } else {
            s.volSpotPipeline = nil
        }
        guard let volSpotUB = device.makeBuffer(length: MemoryLayout<VolSpotUniforms>.stride,
                                                options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("volSpotUniforms")
        }
        volSpotUB.label = "Illuminatorama.volSpot.uniforms"
        s.volSpotUniformBuffer = volSpotUB

        // ── In-view cloud pipeline (opt-in; nil-safe → pass no-ops) ──
        if let cloudFn = library.makeFunction(name: "illumi_cloud_inview") {
            s.cloudInViewPipeline = try? device.makeComputePipelineState(function: cloudFn) // gpu-ok: one-time init, optional in-view cloud pipeline
        } else {
            s.cloudInViewPipeline = nil
        }
    }
}
