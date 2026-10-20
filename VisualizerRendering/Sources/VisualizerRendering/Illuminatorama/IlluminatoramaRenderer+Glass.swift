import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - GlassState

/// Glass (forward RT / fallback) and caustics.
/// Owned by the renderer as `gls`; every member keeps the name it had on the renderer.
@MainActor
final class GlassState {
    var lastGlassFlat: [(kind: IlluminatoramaRenderer.MeshKind, insts: [IlluminatoramaGlassInstance])] = []
    var glassStableFrames: Int = 0

    var glassRTPipeline: MTLRenderPipelineState?  // illumi_glass_rt_fs (traces TLAS)
    var glassFallbackPipeline: MTLRenderPipelineState?  // illumi_glass_fallback_fs (Fresnel+sky)
    var glassDepthState: MTLDepthStencilState?
    /// `rtGlassDefocusTraceFactor` targets: the reduced-resolution trace (rgba16Float, coverage
    /// in alpha) and a same-size depth target cleared to far — the RT glass pipeline carries a
    /// depth attachment, and a cleared one passes every fragment (no occlusion test at low res).
    /// Allocated on first use; nil for every scene that never opts in.
    var glassDefocusTexture: MTLTexture?
    var glassDefocusDepth: MTLTexture?

    /// Per-glass-TLAS-instance material (RTGlassData[], grouped == TLAS glass
    /// order), rebuilt with the TLAS. Indexed by `instance_id - glassInstanceBase`.
    var rtGlassDataBuffer: MTLBuffer?
    /// First glass instance_id in the TLAS (= opaque count + curve count). The RT
    /// glass shader uses this to tell a glass hit from an opaque one.
    var rtGlassInstanceBase: Int = 0
    var rtGlassInstanceCount: Int = 0

    var causticDecayPipeline: MTLComputePipelineState?
    var causticPhotonsPipeline: MTLComputePipelineState?
    var causticPrimaryPipeline: MTLComputePipelineState?
    /// `atomic_float` RGB atlas (3 floats/texel), sized to the surface-cache atlas.
    var causticAtlasBuffer: MTLBuffer?
    /// Per-glass-instance emitter discs (float4 centre+radius), rebuilt per frame.
    var causticDiscBuffer: MTLBuffer?
    var causticAtlasTexels: Int = 0

    var causticFrameSeed: UInt32 = 0

    /// Pre-glass copy of `hdrCompositeTexture`, allocated lazily only when cheap
    /// screen-space glass (mode 2) is in use. The glass pass samples this for the
    /// scene behind the pane (you can't sample a render target you're also writing).
    var glassBackdropTexture: MTLTexture?

    /// Concatenated per-corner SHADING normals of the meshes glass instances use (3 × `float4`
    /// per triangle). A glass instance's slice starts at `RTInstanceData.emissionPad.w − 1`
    /// (0 = none → the bounce loop keeps the face normal). Glass-only on purpose: opaque hits
    /// shade with the face normal and would triple this buffer for nothing.
    var rtGlassCornerNormalBuffer: MTLBuffer?

    /// Per-frame memo for `flattenGlass()` — nil = not yet flattened this frame.
    var glassFlatMemo: [(kind: IlluminatoramaRenderer.MeshKind, insts: [IlluminatoramaGlassInstance])]?
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// Glass that uses DIFFERENT meshes in the same pass. Each group binds its own
    /// mesh and draws its instances; takes precedence over `glassInstances`. Lets
    /// one scene mix glass shapes (a lab row of spheres + prisms + a slab). List
    /// far/background groups first.
    public struct GlassMeshGroup {
        public var kind: MeshKind
        public var instances: [IlluminatoramaGlassInstance]
        public init(kind: MeshKind, instances: [IlluminatoramaGlassInstance]) {
            self.kind = kind
            self.instances = instances
        }
    }

    /// Forward reflective glass pass — one sheet, src-alpha blended over the lit
    /// HDR composite (so its reflections bloom/tonemap), before TAA. No-op unless
    /// the host set a glass pane. Depth-tested against the scene depth (coins in
    /// front occlude) with no depth write (the pile behind shows through).
    /// World-space AABB enclosing all glass instances (unit ±1 object box per
    /// instance transformed to world). Used to aim the caustic photon emitter.
    func glassWorldAABB() -> (SIMD3<Float>, SIMD3<Float>)? {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var any = false
        for (_, insts) in flattenGlass() {
            for inst in insts {
                let m = inst.modelMatrix
                for cx in [-1.0 as Float, 1] { for cy in [-1.0 as Float, 1] { for cz in [-1.0 as Float, 1] {
                    let p = m * SIMD4<Float>(cx, cy, cz, 1)
                    lo = simd_min(lo, SIMD3(p.x, p.y, p.z))
                    hi = simd_max(hi, SIMD3(p.x, p.y, p.z))
                    any = true
                } } }
            }
        }
        return any ? (lo, hi) : nil
    }

    /// Per-glass-instance sun-facing emitter discs (xyz centre, w bounding radius)
    /// — the caustic photon kernel aims a photon at each so ~every photon hits
    /// glass instead of sampling the whole AABB volume (where most miss). Bounding
    /// sphere of the unit ±1 object box under the instance transform.
    func glassEmitterDiscs() -> [SIMD4<Float>] {
        var discs: [SIMD4<Float>] = []
        for (_, insts) in flattenGlass() {
            for inst in insts {
                let m = inst.modelMatrix
                let c = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
                // Half-diagonal of the transformed ±1 box ≈ the bounding radius.
                let ex = simd_length(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z))
                let ey = simd_length(SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z))
                let ez = simd_length(SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
                let radius = simd_length(SIMD3(ex, ey, ez))   // ‖(|col0|,|col1|,|col2|)‖
                discs.append(SIMD4(c, radius))
            }
        }
        return discs
    }

    /// Glass caustics (#60): photon-trace sun → refract through the glass → splat
    /// focused energy into the surface-cache card atlas (any receiver), then read
    /// it back per primary opaque pixel and add to the lit composite. No-op unless
    /// `causticsEnabled` AND the surface cache + glass TLAS are live.
    func encodeCausticPass(_ cb: MTLCommandBuffer) {
        guard causticsEnabled, rts.rtTLASActive, gls.rtGlassInstanceCount > 0,
              let tlas = rts.rtTLAS, let instData = rts.rtInstanceDataBuffer,
              let objN = rts.rtObjNormalBuffer, let gData = gls.rtGlassDataBuffer else { return }
        // Needs the surface-cache card store (this topology) to address receivers.
        let cacheOn = surfCacheActive && scs.surfCardCount > 0
            && rts.rtSoupTriBaseBuffer != nil && scs.surfTriCardBuffer != nil
            && scs.surfTriUVaBuffer != nil && scs.surfTriUVcBuffer != nil && scs.surfCardRectBuffer != nil
            && rts.rtSoupTriBaseCount == instances.count
        guard cacheOn, let aabb = glassWorldAABB() else { return }
        let cache = engine.pipelineCache
        if gls.causticDecayPipeline == nil { gls.causticDecayPipeline = cache.pipelineState(name: "illumi_caustic_decay", device: device) }
        if gls.causticPhotonsPipeline == nil { gls.causticPhotonsPipeline = cache.pipelineState(name: "illumi_caustic_photons", device: device) }
        if gls.causticPrimaryPipeline == nil { gls.causticPrimaryPipeline = cache.pipelineState(name: "illumi_caustic_primary", device: device) }
        guard let decayP = gls.causticDecayPipeline, let photP = gls.causticPhotonsPipeline,
              let primP = gls.causticPrimaryPipeline else { return }
        // Atlas sized to the surface-cache atlas; (re)allocate + clear on growth.
        let texels = scs.surfAtlasW * scs.surfAtlasH
        let needLen = max(1, texels) * 3 * MemoryLayout<Float>.stride
        var freshAtlas = false
        if gls.causticAtlasBuffer == nil || gls.causticAtlasTexels < texels {
            gls.causticAtlasBuffer = device.makeBuffer(length: needLen, options: .storageModePrivate)
            gls.causticAtlasBuffer?.label = "Illuminatorama.caustic.atlas"
            gls.causticAtlasTexels = texels
            freshAtlas = true
        }
        guard let atlas = gls.causticAtlasBuffer else { return }
        if freshAtlas, let blit = cb.makeBlitCommandEncoder() {
            blit.fill(buffer: atlas, range: 0..<needLen, value: 0)   // private buffer can't be memset from CPU
            blit.endEncoding()
        }
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        gls.causticFrameSeed &+= 1
        var u = IlluminatoramaCausticUniforms()
        u.invViewProjection = fu.invViewProjection
        u.cameraWorldPos = fu.cameraWorldPos
        u.rayTMin = 0.004
        u.sunDir = simd_normalize(rtSunDirection)
        u.photonEnergy = max(0, causticPhotonEnergy)
        u.sunColor = rtSunColor
        u.decay = max(0, min(1, causticDecay))
        u.aabbMin = aabb.0; u.aabbMax = aabb.1
        u.strength = max(0, causticStrength)
        u.width = UInt32(width); u.height = UInt32(height)
        u.glassInstanceBase = UInt32(gls.rtGlassInstanceBase)
        u.photonCount = UInt32(max(1, causticPhotonCount))
        u.surfTriCount = UInt32(rts.rtTriangleCount)
        u.surfAtlasW = UInt32(scs.surfAtlasW); u.surfAtlasH = UInt32(scs.surfAtlasH)
        u.frameSeed = gls.causticFrameSeed
        u.maxGlassBounces = UInt32(max(2, min(8, rtGlassMaxBounces)))
        // Per-instance emitter discs — aim photons at glass directly (near-100% hit).
        let discs = glassEmitterDiscs()
        let discLen = max(1, discs.count) * MemoryLayout<SIMD4<Float>>.stride
        if (gls.causticDiscBuffer?.length ?? 0) < discLen {
            gls.causticDiscBuffer = device.makeBuffer(length: discLen, options: .storageModeShared)
            gls.causticDiscBuffer?.label = "Illuminatorama.caustic.discs"
        }
        if let db = gls.causticDiscBuffer, !discs.isEmpty {
            discs.withUnsafeBytes { _ = memcpy(db.contents(), $0.baseAddress!, discs.count * MemoryLayout<SIMD4<Float>>.stride) }
        }
        u.glassDiscCount = UInt32(discs.count)
        memcpy(causticUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaCausticUniforms>.stride)

        // Pass 1 — temporal decay.
        if let enc = timedComputeEncoder(cb, "caustic.decay") {
            enc.label = "Illuminatorama.caustic.decay"
            enc.setComputePipelineState(decayP)
            enc.setBuffer(atlas, offset: 0, index: 0)
            enc.setBuffer(causticUniformBuffer, offset: 0, index: 1)
            dispatch1D(enc, pipeline: decayP, count: texels * 3)
            enc.endEncoding()
        }
        // Pass 2 — emit photons, refract through glass, splat into receiver cards.
        if let enc = timedComputeEncoder(cb, "caustic.photons") {
            enc.label = "Illuminatorama.caustic.photons"
            enc.setComputePipelineState(photP)
            enc.setBuffer(atlas, offset: 0, index: 0)
            enc.setBuffer(causticUniformBuffer, offset: 0, index: 1)
            enc.setAccelerationStructure(tlas, bufferIndex: 2)
            enc.setBuffer(instData, offset: 0, index: 3)
            enc.setBuffer(objN, offset: 0, index: 4)
            enc.setBuffer(gData, offset: 0, index: 5)
            enc.setBuffer(rts.rtSoupTriBaseBuffer, offset: 0, index: 6)
            enc.setBuffer(scs.surfTriCardBuffer, offset: 0, index: 7)
            enc.setBuffer(scs.surfTriUVaBuffer, offset: 0, index: 8)
            enc.setBuffer(scs.surfTriUVcBuffer, offset: 0, index: 9)
            enc.setBuffer(scs.surfCardRectBuffer, offset: 0, index: 10)
            enc.setBuffer(gls.causticDiscBuffer ?? causticUniformBuffer, offset: 0, index: 11)
            for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
            for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
            dispatch1D(enc, pipeline: photP, count: max(1, causticPhotonCount))
            enc.endEncoding()
        }
        // Pass 3 — per primary opaque pixel, read its own card's caustic + add.
        if let enc = timedComputeEncoder(cb, "caustic.primary") {
            enc.label = "Illuminatorama.caustic.primary"
            enc.setComputePipelineState(primP)
            enc.setTexture(gbs.depthTexture, index: 0)
            enc.setTexture(gbs.gbufferAlbedoMet, index: 1)
            enc.setTexture(pfx.hdrCompositeTexture, index: 2)
            enc.setBuffer(causticUniformBuffer, offset: 0, index: 0)
            enc.setBuffer(atlas, offset: 0, index: 1)
            enc.setAccelerationStructure(tlas, bufferIndex: 2)
            enc.setBuffer(rts.rtSoupTriBaseBuffer, offset: 0, index: 3)
            enc.setBuffer(scs.surfTriCardBuffer, offset: 0, index: 4)
            enc.setBuffer(scs.surfTriUVaBuffer, offset: 0, index: 5)
            enc.setBuffer(scs.surfTriUVcBuffer, offset: 0, index: 6)
            enc.setBuffer(scs.surfCardRectBuffer, offset: 0, index: 7)
            for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
            for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
            dispatch(enc, pipeline: primP, width: width, height: height,
                     maxThreadsOverride: rtThreadgroupMax)
            enc.endEncoding()
        }
    }

    /// Canonical glass draw list — the SINGLE flattening both the raster pass and
    /// the TLAS build use, so a glass hit's material (looked up by TLAS instance
    /// order) matches what the raster fragment drew. Precedence: multi-mesh
    /// groups → single-mesh array → single pane.
    func flattenGlass() -> [(kind: MeshKind, insts: [IlluminatoramaGlassInstance])] {
        // Memoised per frame (invalidated at the top of `render()`): the topology
        // hash, TLAS refit, glass pass, and caustics AABB all call this in one
        // frame, and hosts only mutate glass between frames.
        if let memo = gls.glassFlatMemo { return memo }
        let flat: [(kind: MeshKind, insts: [IlluminatoramaGlassInstance])]
        if !glassMeshGroups.isEmpty {
            flat = glassMeshGroups.compactMap { $0.instances.isEmpty ? nil : ($0.kind, $0.instances) }
        } else if let kind = glassPaneKind, !glassInstances.isEmpty {
            flat = [(kind, glassInstances)]
        } else if let kind = glassPaneKind, let single = glassPaneInstance {
            flat = [(kind, [single])]
        } else {
            flat = []
        }
        gls.glassFlatMemo = flat
        return flat
    }

    func encodeGlassPass(_ cb: MTLCommandBuffer) {
        guard let glassDepth = gls.glassDepthState else { return }
        let groups = flattenGlass()
        guard !groups.isEmpty else { return }

        // RT path when the device + scene support a live TLAS that holds this
        // frame's glass; otherwise the Fresnel-only fallback. `rtGlassInstanceCount`
        // is set by `rebuildRTAccel` when glass was appended to the TLAS.
        let useRT = rtGlassEnabled && rts.rtTLASActive && gls.rtGlassInstanceCount > 0
            && gls.glassRTPipeline != nil && rts.rtTLAS != nil
            && rts.rtInstanceDataBuffer != nil && rts.rtObjNormalBuffer != nil
            && gls.rtGlassDataBuffer != nil
        guard let pipeline = useRT ? gls.glassRTPipeline : gls.glassFallbackPipeline else { return }

        // Pack every group's instances contiguously; each group draws from its own
        // byte offset (one CPU upload feeds all draws).
        let stride = MemoryLayout<IlluminatoramaGlassInstance>.stride
        var flat: [IlluminatoramaGlassInstance] = []
        var offsets: [Int] = []
        for g in groups { offsets.append(flat.count); flat.append(contentsOf: g.insts) }
        let needed = stride * flat.count
        if glassInstanceBuffer.length < needed {
            guard let grown = device.makeBuffer(length: needed,
                                                options: .storageModeShared) else { return }
            glassInstanceBuffer = grown
        }
        flat.withUnsafeBytes { src in
            _ = memcpy(glassInstanceBuffer.contents(), src.baseAddress!, needed)
        }

        // Per-frame RT-glass uniforms (camera + RT sun/ambient + bases + flags).
        let fu = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self)
        rts.rtInstFrameSeed &+= 1
        // Surface-cache availability (mirror of the TLAS lighting pass): a glass
        // refraction/reflection hit reads the cached multi-bounce radiance when on.
        let cacheOn = useRT && surfCacheActive && scs.surfCardCount > 0
            && rts.rtSoupTriBaseBuffer != nil && scs.surfTriCardBuffer != nil
            && surfConsumerAtlas != nil
            && rts.rtSoupTriBaseCount == instances.count
        var u = IlluminatoramaGlassRTUniforms()
        u.cameraWorldPos = fu.cameraWorldPos
        u.rayTMin = 0.004
        u.sunDir = simd_normalize(rtSunDirection)
        u.sunSoftnessRad = max(0.0005, rtSunSoftnessRad)
        u.sunColor = rtSunColor
        u.reflStrength = max(0.0001, rtReflStrength)
        u.skyAmbient = ambientColor
        u.skyIntensity = max(0, iblIntensity)
        u.glassInstanceBase = UInt32(gls.rtGlassInstanceBase)
        u.maxBounces = max(2, min(10, rtGlassMaxBounces))
        u.shadowRays = max(0, min(8, rtGlassShadowRays))
        // The glass pass samples its sun shadow stochastically (one cone ray by default) and
        // decorrelates the sequence per frame so that TAA can average it back into a smooth
        // penumbra — `rtGlassShadowRays`' own docs say "TAA cleans the penumbra". When TAA is
        // NOT resolving, nothing ever averages it, and a walking seed turns that sampling noise
        // into speckle that CRAWLS on a completely static frame: the camera has stopped, the
        // scene has stopped, and the trim around a window still boils. Freeze the seed in that
        // case. The sampling error is identical in magnitude — it just stops moving, which is
        // the difference between a fixed dither the eye reads as texture and a defect. Strict
        // no-op for any scene running TAA.
        u.frameSeed = taaEnabled ? rts.rtInstFrameSeed : 0
        u.surfCacheEnabled = cacheOn ? 1 : 0
        u.surfTriCount = cacheOn ? UInt32(rts.rtTriangleCount) : 0
        u.surfAtlasW = cacheOn ? UInt32(scs.surfAtlasW) : 0
        u.surfAtlasH = cacheOn ? UInt32(scs.surfAtlasH) : 0
        u.dispersionEnabled = rtGlassDispersionEnabled ? 1 : 0
        // Graceful RT-glass degradation (issue #65): a scene that asked for RT glass
        // (`rtGlassEnabled`) but whose TLAS isn't live this frame — unsupported
        // device, not built yet, or no glass in the TLAS — auto-falls back to the
        // best screen-space cheap glass (mode 2, refractive + thin-film) instead of
        // the flat Fresnel-only fallback, with NO per-scene `glassCheapMode` wiring.
        // An explicit `glassCheapMode` still wins if a host set one.
        let autoCheapFallback = rtGlassEnabled && !useRT
        let effectiveCheapMode = glassCheapMode != 0 ? glassCheapMode
                               : (autoCheapFallback ? 2 : 0)
        // Cheap glass: only on the non-RT path, and only if opted in OR auto-fallback.
        let cheap = !useRT && effectiveCheapMode != 0
        u.cheapGlassMode = cheap ? UInt32(effectiveCheapMode) : 0
        // Screen-space transmission parity is an RT-path feature: the cheap paths already
        // read the composite (mode 2) or have no transmitted lobe to speak of (modes 0/1).
        let ssTransmissionOn = useRT && rtGlassScreenSpaceTransmission
        u.ssTransmissionEnabled = ssTransmissionOn ? 1 : 0
        u.viewW = Float(pfx.hdrCompositeTexture.width)
        u.viewH = Float(pfx.hdrCompositeTexture.height)
        // Thin-film iridescence (cheap mode 2 only; default strength 0 = no-op).
        u.time = time
        u.thinFilmStrength = (cheap && effectiveCheapMode == 2) ? max(0, thinFilmStrength) : 0
        u.filmThicknessNm = max(1, thinFilmThicknessNm)
        u.filmIOR = max(1.01, thinFilmIOR)
        // Oscillation-mode undulation — only on the cheap path (bubble shells).
        u.wobbleAmp = cheap ? max(0, bubbleWobbleAmp) : 0
        u.wobbleFreq = max(0.01, bubbleWobbleFreq)
        // ── Through-glass parity with the deferred pass (RT path only) ───────
        // A refracted ray's hit used to be shaded by a strictly poorer model than
        // the same surface one pixel to the left: mean albedo, no texture, no local
        // lights, exterior-strength fill. These three give it the same inputs.
        // …each independently suppressible, so the three can be ABLATED against a gate. They
        // landed together, which made the shipped number (detail through a pane, vs the same
        // view with the glazing removed) unattributable — and an unattributed win teaches
        // nothing about where to spend the next hour on the sibling RT paths. Ship-path
        // default for all three is `true`; only a harness sets them.
        let rtShade = useRT
        u.albedoAtlasEnabled = (rtShade && glassHitTexturingEnabled
                                && rts.rtObjUVBuffer != nil && rts.rtObjUVCount > 0) ? 1 : 0
        u.objUVCount = UInt32(rts.rtObjUVCount)
        u.pointLightCount = (rtShade && glassHitLocalLightsEnabled) ? UInt32(pointLights.count) : 0
        u.spotLightCount = (rtShade && glassHitLocalLightsEnabled) ? UInt32(spotLights.count) : 0
        u.interiorMask = (rtShade && glassHitInteriorFillEnabled) ? interiorLayerMask : 0
        u.interiorIBLUp = max(0, interiorIBLUp)
        u.interiorIBLSide = max(0, interiorIBLSide)
        u.interiorAmbient = max(0, interiorAmbient)
        // Interior irradiance bands — the room seen THROUGH a pane must agree with
        // the room beside it (the lawn-green-ceiling fix, glass side).
        let glassIrrW = max(0, min(1, interiorIrradianceWeight))
        u.interiorIrrUp = SIMD4(simd_max(interiorIrradianceUp, .zero), glassIrrW)
        u.setInteriorRoomGains(interiorRoomGains, enabled: interiorRoomGainsEnabled)
        // DH-0718 — area lights at through-glass hits too (same body, same lanes).
        u.interiorRoomGainMeta.y = rtShade ? Float(secondaryAreaLightCount) : 0
        u.interiorRoomGainMeta.z = Float(max(0, min(secondaryAreaShadowRays, 8)))
        u.interiorIrrSide = SIMD4(simd_max(interiorIrradianceSide, .zero), 0)
        u.interiorIrrDown = SIMD4(simd_max(interiorIrradianceDown, .zero), 0)
        // ── The night sky, THROUGH the glass ─────────────────────────────────
        // Same fields the deferred sky branch reads, from the same properties, so a
        // window and the sky beside it show one sky. Without this the pane sampled the
        // celestial-free dome and a night window rendered flat black while the open sky
        // carried the whole star field. All-zero (any non-night host) ⇒ shader no-op.
        u.nightSkyParams = SIMD4(max(0, nightSkyStarBrightness),
                                 max(0, nightSkyMoonIntensity),
                                 max(0, nightSkyMoonAngularRadius), 0)
        u.nightMoonDir = SIMD4(nightSkyMoonDirection == .zero
                               ? SIMD3<Float>(0, 1, 0) : simd_normalize(nightSkyMoonDirection), 0)
        u.nightSunDir = SIMD4(nightSkySunDirection == .zero
                              ? SIMD3<Float>(0, -1, 0) : simd_normalize(nightSkySunDirection), 0)
        (u.nightSkyExtra, u.nightSkyExtra2, u.nightCelestial) = physicalNightSkyClusters()
        // One pixel's angular size, derived exactly as the primary branch derives it
        // (tan(fovY/2) = 1/P[1][1], over the render height) so a star is the same size
        // seen directly and seen through a pane.
        //
        // Perspective only. Under a parallel projection P[1][1] is 1/halfHeight, so
        // `2/P[1][1]` is a LENGTH in metres rather than an angle — feeding that here
        // would scale every star by the frame's world size (a 40 m axonometric frame
        // would draw stars ~100× oversized). A parallel camera has no per-pixel angle
        // to speak of, so fall back to the stored `fovYRadians`: unused for framing in
        // that mode, but still the honest "what lens is this roughly" value.
        let pixExtent = camera.isOrthographic
            ? 2 * tan(camera.fovYRadians * 0.5)
            : 2.0 / max(camera.projectionMatrix[1][1], 1e-6)
        u.nightPixAngle = pixExtent / Float(max(pfx.hdrCompositeTexture.height, 1))

        // ── Defocus trace (opt-in `rtGlassDefocusTraceFactor`, VZ-0197) ──────────────
        // The full-res pass (mode 2) and the reduced-res pass (mode 1) share every binding but
        // their uniforms: the low-res copy has the low-res viewport (backdrop UV) and pixel angle
        // (a star's point spread is still one of ITS pixels — the flux is integrated over the
        // footprint, so the reconstruction conserves it). The lens fields mirror the DOF pass's
        // own clamps exactly, so the gate agrees with the blur the DOF then applies.
        let defocusK = max(1, min(4, rtGlassDefocusTraceFactor))
        let defocusOn = useRT && defocusK > 1 && dofEnabled && dofCoCCoefficient > 0
            && !camera.isOrthographic && ensureGlassDefocusTargets(k: defocusK)
        var lowU = u
        if defocusOn {
            let k = Float(defocusK)
            let gate = max(0.5, rtGlassDefocusMinCoCTexels) * k
            let axis = camera.target - camera.position
            let forward = simd_length(axis) > 1e-6 ? simd_normalize(axis) : SIMD3<Float>(0, 0, -1)
            u.defocus = SIMD4(2, k, gate, gate * 0.75)
            u.defocusLens = SIMD4(max(0.05, dofFocusDistance), max(0, dofCoCCoefficient),
                                  max(1, dofMaxRadius), max(0, dofDiffractionCoCPixels))
            u.cameraForward = SIMD4(forward, 0)
            lowU = u
            lowU.defocus.x = 1
            lowU.viewW = u.viewW / k
            lowU.viewH = u.viewH / k
            lowU.nightPixAngle = u.nightPixAngle * k
        }
        memcpy(glassRTUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaGlassRTUniforms>.stride)

        // Screen-space cheap glass (mode 2) samples the scene BEHIND the pane: copy
        // the pre-glass composite into a backdrop texture (can't sample the render
        // target we're about to write). One full-frame GPU blit; no ray tracing.
        // The RT path needs it too now: screen-space transmission parity samples the
        // pre-glass composite for the transmitted lobe at near-normal incidence.
        let useBackdrop = (cheap && effectiveCheapMode == 2) || ssTransmissionOn
        if useBackdrop {
            if gls.glassBackdropTexture?.width != pfx.hdrCompositeTexture.width
                || gls.glassBackdropTexture?.height != pfx.hdrCompositeTexture.height {
                let d = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: pfx.hdrCompositeTexture.pixelFormat,
                    width: pfx.hdrCompositeTexture.width, height: pfx.hdrCompositeTexture.height,
                    mipmapped: false)
                d.usage = [.shaderRead]; d.storageMode = .private
                gls.glassBackdropTexture = device.makeTexture(descriptor: d)
                gls.glassBackdropTexture?.label = "Illuminatorama.glassBackdrop"
            }
            if let bd = gls.glassBackdropTexture, let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: pfx.hdrCompositeTexture, to: bd)
                blit.endEncoding()
            }
        }

        /// One glass draw pass's state, bindings and draws. `lowRes` = the defocus trace's
        /// reduced-resolution pass: its uniforms go in by value, and the low-res result is not
        /// bound to itself (the full-res pass reads it at fragment texture 5).
        func encodeGlassDraws(_ enc: MTLRenderCommandEncoder, lowRes: Bool,
                              uniforms: inout IlluminatoramaGlassRTUniforms) {
            enc.setRenderPipelineState(pipeline)
            enc.setDepthStencilState(glassDepth)
            // Perf: the RT path's bounce loop traces the WHOLE glass volume from the
            // front (entry) surface, so back faces are wasted invocations (overwritten
            // by the nearer front face under depth-LE + no-write). Cull them — ~half
            // the glass fragments on convex glass, identical output. (CCW-front matches
            // the G-buffer winding.) The Fresnel+sky FALLBACK still wants both surfaces,
            // so it stays two-sided.
            if useRT || (cheap && effectiveCheapMode == 2) {
                // Screen-space cheap glass traces from the FRONT pane (samples the scene
                // behind it), so back faces are wasted/double-darkening — cull them.
                enc.setCullMode(.back)
                enc.setFrontFacing(.counterClockwise)
            } else {
                enc.setCullMode(.none)                        // two-sided glass (fallback / synthetic)
            }
            enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)   // viewProjection is first
            if lowRes {
                let len = MemoryLayout<IlluminatoramaGlassRTUniforms>.stride
                enc.setVertexBytes(&uniforms, length: len, index: 3)
                enc.setFragmentBytes(&uniforms, length: len, index: 1)
            } else {
                enc.setVertexBuffer(glassRTUniformBuffer, offset: 0, index: 3) // VS reads time/wobble for oscillation-mode undulation
                enc.setFragmentBuffer(glassRTUniformBuffer, offset: 0, index: 1)
            }
            // The defocus trace's low-res result (full-res pass in defocus mode only); any texture
            // otherwise — the shader reads it only in mode 2.
            enc.setFragmentTexture((!lowRes && defocusOn ? gls.glassDefocusTexture : nil) ?? ibls.dummySkyTexture, index: 5)
            enc.setFragmentTexture(equirectSky ?? ibls.dummySkyTexture, index: 0)
            // Backdrop (cheap mode 2) — bind a dummy otherwise so the slot is valid.
            enc.setFragmentTexture((useBackdrop ? gls.glassBackdropTexture : nil) ?? ibls.dummySkyTexture, index: 2)
            if useRT, let tlas = rts.rtTLAS, let instData = rts.rtInstanceDataBuffer,
               let objN = rts.rtObjNormalBuffer, let gData = gls.rtGlassDataBuffer {
                enc.setFragmentAccelerationStructure(tlas, bufferIndex: 0)
                enc.setFragmentBuffer(instData, offset: 0, index: 3)
                enc.setFragmentBuffer(objN, offset: 0, index: 4)
                enc.setFragmentBuffer(gData, offset: 0, index: 5)
                // Surface-cache buffers (dummies keep the bindings valid when off);
                // optional-ternary mirrors the TLAS lighting pass — no force-unwrap.
                let dummy: MTLBuffer = instData
                enc.setFragmentBuffer(cacheOn ? scs.surfTriCardBuffer : dummy, offset: 0, index: 6)
                enc.setFragmentBuffer(cacheOn ? scs.surfTriUVaBuffer : dummy, offset: 0, index: 7)
                enc.setFragmentBuffer(cacheOn ? scs.surfTriUVcBuffer : dummy, offset: 0, index: 8)
                enc.setFragmentBuffer(cacheOn ? rts.rtSoupTriBaseBuffer : dummy, offset: 0, index: 9)
                enc.setFragmentBuffer(cacheOn ? scs.surfCardRectBuffer : dummy, offset: 0, index: 10)
                enc.setFragmentBuffer(cacheOn ? scs.surfCardBuffer : (surfCardDummyBuffer ?? dummy), offset: 0, index: 11)
                enc.setFragmentTexture(cacheOn ? surfConsumerAtlas : (equirectSky ?? ibls.dummySkyTexture), index: 1)
                // Through-glass world parity: re-shaded opaque hits sample the SAME
                // cosine-convolved irradiance cube the deferred lighting pass uses for
                // its diffuse IBL, so transmitted scenery gets the sky fill (not just
                // the flat ambient supplement) and matches the world beside the pane.
                enc.setFragmentTexture(ibls.irradianceCube, index: 3)
                // Through-glass TEXTURE parity: the per-triangle mesh UVs + the albedo
                // atlas (and its per-slice letterbox table), so a re-shaded hit samples
                // the SAME texel the G-buffer would instead of the instance's mean
                // albedo. Without these the world behind a pane is flat slabs.
                enc.setFragmentBuffer(rts.rtObjUVBuffer ?? instData, offset: 0, index: 12)
                enc.setFragmentBuffer(albedoAtlas.uvScaleBuffer, offset: 0, index: 13)
                enc.setFragmentTexture(albedoAtlas.texture, index: 4)
                // Local lights: the deferred kernel accumulates point + spot; the glass
                // pass used to have sun-only, so a lamp-lit room went black through a
                // window. Same ring buffers the lighting kernel reads this frame.
                enc.setFragmentBuffer(pointLightBuffer, offset: 0, index: 14)
                enc.setFragmentBuffer(spotLightBuffer, offset: 0, index: 15)
                enc.setFragmentBuffer(shd.spotProfileBuffer, offset: 0, index: 18)   // DH-0540 — the spot-profile rows
                // Smooth refraction: glass hits interpolate these corner normals (a dummy with
                // every row's base 0 means "face normal" — see `rtGlassCornerNormalBuffer`).
                enc.setFragmentBuffer(gls.rtGlassCornerNormalBuffer ?? instData, offset: 0, index: 16)
                // DH-0718 — area lights at the refracted hit (buffer 17); dummy when none.
                enc.setFragmentBuffer(secondaryAreaLightCount > 0 ? areaLightBuffer : instData,
                                      offset: 0, index: 17)
                // The TLAS references the BLASes which reference mesh buffers — all
                // must be resident for the fragment-stage intersector.
                for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
                for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
            }
            for (i, g) in groups.enumerated() {
                guard let mesh = meshes[g.kind] else { continue }
                let off = stride * offsets[i]
                enc.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
                enc.setVertexBuffer(glassInstanceBuffer, offset: off, index: 2)
                enc.setFragmentBuffer(glassInstanceBuffer, offset: off, index: 2)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: mesh.indexCount,
                                          indexType: mesh.indexType, indexBuffer: mesh.indexBuffer,
                                          indexBufferOffset: 0, instanceCount: g.insts.count)
            }
        }

        // The low-res trace first (defocus only): its own targets, no occlusion test, its own
        // uniforms (by value — the shared uniform buffer holds the full-res pass's).
        if defocusOn, let loTex = gls.glassDefocusTexture, let loDepth = gls.glassDefocusDepth {
            let lp = MTLRenderPassDescriptor()
            lp.colorAttachments[0].texture = loTex
            lp.colorAttachments[0].loadAction = .clear   // coverage 0 wherever no glass traces
            lp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            lp.colorAttachments[0].storeAction = .store
            lp.depthAttachment.texture = loDepth
            lp.depthAttachment.loadAction = .clear       // far: every fragment passes
            lp.depthAttachment.clearDepth = 1.0
            lp.depthAttachment.storeAction = .dontCare
            if let lenc = timedRenderEncoder(cb, lp, "glass.lowres") {
                lenc.label = "Illuminatorama.glass.rt.defocusLowRes"
                let k = Double(defocusK)
                lenc.setViewport(MTLViewport(originX: 0, originY: 0,
                                             width: Double(pfx.hdrCompositeTexture.width) / k,
                                             height: Double(pfx.hdrCompositeTexture.height) / k,
                                             znear: 0, zfar: 1))
                encodeGlassDraws(lenc, lowRes: true, uniforms: &lowU)
                lenc.endEncoding()
            }
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = pfx.hdrCompositeTexture
        pass.colorAttachments[0].loadAction = .load     // keep the lit scene
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = gbs.depthTexture
        pass.depthAttachment.loadAction = .load          // test vs scene depth
        pass.depthAttachment.storeAction = .dontCare     // no depth write
        guard let enc = timedRenderEncoder(cb, pass, "glass") else { return }
        enc.label = useRT ? "Illuminatorama.glass.rt"
            : (cheap ? "Illuminatorama.glass.cheap\(effectiveCheapMode)\(autoCheapFallback && glassCheapMode == 0 ? "-auto" : "")" : "Illuminatorama.glass.fallback")
        encodeGlassDraws(enc, lowRes: false, uniforms: &u)
        enc.endEncoding()
    }

    /// (Re)allocate the defocus trace's targets for factor `k` at the current internal size.
    /// False (the caller then traces every fragment in full) if an allocation fails.
    func ensureGlassDefocusTargets(k: Int) -> Bool {
        let w = max(1, (pfx.hdrCompositeTexture.width + k - 1) / k)
        let h = max(1, (pfx.hdrCompositeTexture.height + k - 1) / k)
        if gls.glassDefocusTexture?.width != w || gls.glassDefocusTexture?.height != h {
            let c = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h,
                                                             mipmapped: false)
            c.usage = [.renderTarget, .shaderRead]
            c.storageMode = .private
            gls.glassDefocusTexture = device.makeTexture(descriptor: c)
            gls.glassDefocusTexture?.label = "Illuminatorama.glass.defocusLowRes"
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: w, height: h,
                                                             mipmapped: false)
            d.usage = [.renderTarget]
            d.storageMode = .private
            gls.glassDefocusDepth = device.makeTexture(descriptor: d)
            gls.glassDefocusDepth?.label = "Illuminatorama.glass.defocusLowResDepth"
        }
        return gls.glassDefocusTexture != nil && gls.glassDefocusDepth != nil
    }
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// AAA glass pipelines: RT variant (RT-capable devices only), Fresnel fallback, depth state.
    @MainActor static func buildGlassPipelines(into s: GlassState, device: MTLDevice, library: MTLLibrary) {
        // ── AAA glass pipelines (#60, forward, src-alpha blended) ──────────
        // Two fragment variants share one vertex stage (`illumi_glass_rt_vs`) and
        // one blend config: `illumi_glass_rt_fs` traces the scene TLAS for true
        // refraction/reflection (returns opaque alpha 1 → nearest pane replaces),
        // `illumi_glass_fallback_fs` is the Fresnel+sky fallback (alpha < 1 →
        // blends). Drawn into the HDR composite after lighting, depth-tested LE but
        // no depth write so opaque geometry in front occludes the glass.
        // Local-capture closure (a nested `func` would implicitly capture `self`
        // before full init); `dev`/`library` are locals so this is init-safe.
        let dev = device
        let makeGlassPipeline: (String, String) -> MTLRenderPipelineState? = { fragName, label in
            guard let gvs = library.makeFunction(name: "illumi_glass_rt_vs"),
                  let gfs = library.makeFunction(name: fragName) else { return nil }
            let gdesc = MTLRenderPipelineDescriptor()
            gdesc.label = label
            gdesc.vertexFunction = gvs
            gdesc.fragmentFunction = gfs
            gdesc.colorAttachments[0].pixelFormat = .rgba16Float   // hdrCompositeTexture
            gdesc.colorAttachments[0].isBlendingEnabled = true
            gdesc.colorAttachments[0].rgbBlendOperation = .add
            gdesc.colorAttachments[0].alphaBlendOperation = .add
            gdesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            gdesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            gdesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            gdesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            gdesc.depthAttachmentPixelFormat = .depth32Float
            return try? dev.makeRenderPipelineState(descriptor: gdesc)
        }
        // The RT fragment shader needs a device that supports ray tracing; build
        // it only there (the function references `instance_acceleration_structure`).
        s.glassRTPipeline = device.supportsRaytracing
            ? makeGlassPipeline("illumi_glass_rt_fs", "Illuminatorama.glass.rt") : nil
        s.glassFallbackPipeline = makeGlassPipeline("illumi_glass_fallback_fs",
                                                       "Illuminatorama.glass.fallback")
        let gds = MTLDepthStencilDescriptor()
        gds.depthCompareFunction = .lessEqual
        gds.isDepthWriteEnabled = false
        s.glassDepthState = device.makeDepthStencilState(descriptor: gds)
    }
}
