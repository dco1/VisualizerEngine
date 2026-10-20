import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - Frame / init steps extracted from the main file

extension IlluminatoramaRenderer {
    /// Re-prime each temporal history on the frame its feature turns (back) on, or the camera/representation changes. Called once per `render()`.
    func primeTemporalHistoriesOnTransitions() {
        // Track enable-transitions so a re-enable re-primes each temporal history.
        if taaEnabled && !pfx.previousTaaEnabled { pfx.taaNeedsFirstFrame = true }
        pfx.previousTaaEnabled = taaEnabled
        // DH-0887 — the frozen-camera resolve reads the same pixel's history with no reprojection
        // and no clip, so a CAMERA CUT inside a still (an angle sheet stepping yaw/pitch in one
        // capture, a Go To mid-export) must re-prime it, or every shot double-exposes the last.
        // Keyed on the UNJITTERED view-projection: the jitter moves every frame on purpose.
        // Entering STILL mode for the GI restarts its mean: the history it would continue is a
        // moving-camera EMA (different weights, clamped), not a sum of this picture's samples.
        if rtGIStaticAccumulation && !rts.previousRTGIStatic { rtGINeedsFirstFrame = true }
        rts.previousRTGIStatic = rtGIStaticAccumulation
        // …and so does any switch between the two representations (modulated ↔ irradiance): the history
        // the other one wrote is not a history of this one.
        if giIrradianceMode != rts.previousGIIrradianceMode { rtGINeedsFirstFrame = true }
        rts.previousGIIrradianceMode = giIrradianceMode
        if taaStillAccumulation {
            let vp = camera.projectionMatrix * camera.viewMatrix
            if let last = rts.lastStillViewProjection, last != vp { pfx.taaNeedsFirstFrame = true }
            rts.lastStillViewProjection = vp
        } else {
            rts.lastStillViewProjection = nil
        }
        if ssaoTemporalEnabled && !aos.previousSsaoEnabled { aos.aoNeedsFirstFrame  = true }
        aos.previousSsaoEnabled = ssaoTemporalEnabled
        // SSAO is skipped entirely while `ssaoIntensity <= 0` (the chain gate
        // below), so its temporal history goes stale during the off period —
        // re-prime it when intensity comes back, exactly as SSR does.
        if ssaoActive && !aos.previousSSAOIntensityActive { aos.aoNeedsFirstFrame = true }
        aos.previousSSAOIntensityActive = ssaoActive
        if ssrTemporalEnabled && !ssrs.previousSsrEnabled  { ssrs.ssrNeedsFirstFrame = true }
        ssrs.previousSsrEnabled  = ssrTemporalEnabled
        // SSR is skipped entirely while `ssrIntensity <= 0` (see the gather/temporal/
        // composite gate below), so its temporal history goes stale during the off
        // period — re-prime it when intensity comes back.
        if ssrIntensity > 0 && !ssrs.previousSSRIntensityActive { ssrs.ssrNeedsFirstFrame = true }
        ssrs.previousSSRIntensityActive = ssrIntensity > 0
        if rtGITemporalEnabled && !rts.previousRTGITemporalEnabled { rtGINeedsFirstFrame = true }
        rts.previousRTGITemporalEnabled = rtGITemporalEnabled
    }

    /// Scene-content stability update (perf gate — see the state block): once per RENDERED frame, after the in-flight guard and before the uploads.
    func updateSceneStability() {
        // ── Scene-content stability update (perf gate — see the state block) ──
        // Runs once per RENDERED frame (after the in-flight guard, before the
        // uploads), so a stable count of N means N consecutive uploads saw this
        // exact content. Glass is flattened once here and memoised for the frame
        // (rtTopologyHash / refit / glass pass / caustics all reuse it instead of
        // re-flattening 4–5×).
        gls.glassFlatMemo = nil
        let glassNow = flattenGlass()
        if instances == gbs.lastUploadedInstances {
            gbs.instanceStableFrames += 1
            gbs.instanceShapeStableFrames += 1
        } else {
            // Full content changed. Distinguish "geometry moved" from "only a
            // material/emission animated" (the canonical case: a powered-on TV's
            // screen flicker rewrites emission+albedo every frame). The shadow
            // depth passes read ONLY meshKind + modelMatrix + the sway fields,
            // so SHAPE stability keeps the up-to-32-pass shadow reuse alive
            // under animated materials. Compared per-field only on full-compare
            // misses, so the static-scene fast path never pays for it.
            let shapeHeld = instances.count == gbs.lastUploadedInstances.count
                && zip(instances, gbs.lastUploadedInstances).allSatisfy {
                    $0.meshKind == $1.meshKind
                        && $0.data.modelMatrix == $1.data.modelMatrix
                        && $0.data.swayMode == $1.data.swayMode
                        && $0.data.swayLean == $1.data.swayLean
                        && $0.data.swayJostle == $1.data.swayJostle
                }
            gbs.instanceShapeStableFrames = shapeHeld ? gbs.instanceShapeStableFrames + 1 : 0
            gbs.instanceStableFrames = 0
            gbs.lastUploadedInstances = instances   // COW reference, no deep copy
            gbs.contentHasSway = instances.contains { $0.data.swayMode >= 2 }
            rebuildSwayingSpheres()
        }
        if glassNow.count == gls.lastGlassFlat.count,
           zip(glassNow, gls.lastGlassFlat).allSatisfy({ $0.kind == $1.kind && $0.insts == $1.insts }) {
            gls.glassStableFrames += 1
        } else {
            gls.glassStableFrames = 0
            gls.lastGlassFlat = glassNow
        }
        // Kill-switch: zeroing the counters here disables every downstream skip
        // (upload / TLAS refit / shadow reuse) in one place.
        if Self.staticSkipDisabled {
            gbs.instanceStableFrames = 0
            gbs.instanceShapeStableFrames = 0
            gls.glassStableFrames = 0
        }
    }

    /// Bookkeeping for next frame: previous VP/jitter, TAA index, and each temporal history ping-pong that actually ran.
    func advanceTemporalHistories() {
        // Bookkeeping for next frame:
        // - Save this frame's jittered VP as the previous VP next time.
        // - Advance the jitter index so the Halton sequence walks forward.
        // - We've now written a valid history frame.
        pfx.previousViewProjection = pfx.lastFrameViewProjection
        pfx.previousJitterNDC = pfx.lastFrameJitterNDC
        if taaEnabled {
            pfx.taaFrameIndex &+= 1
            pfx.taaNeedsFirstFrame = false
            pfx.historyToggle.toggle()
        }
        // Only when SSAO actually RAN this frame. While it's skipped
        // (`ssaoIntensity <= 0`) the history must neither toggle nor clear its
        // first-frame flag — otherwise the ping-pong would keep advancing over
        // two buffers nothing is writing, and a later re-enable would blend
        // against arbitrarily old occlusion. Same contract as the SSR chain.
        if ssaoTemporalEnabled && ssaoChainEncodes {
            aos.aoNeedsFirstFrame = false
            aos.aoHistoryToggle.toggle()
        }
        // Only when SSR actually RAN this frame (intensity > 0): while it's
        // skipped, the history must neither toggle nor clear its first-frame
        // flag, so a later re-enable re-primes from a clean slate.
        if ssrTemporalEnabled && ssrIntensity > 0 {
            ssrs.ssrNeedsFirstFrame = false
            ssrs.ssrHistoryToggle.toggle()
        }
        if rtGITemporalEnabled && rtEnabled && rtSupported {
            rtGINeedsFirstFrame = false
            rts.rtGIHistoryToggle.toggle()
        }
    }

    /// Adopt the freshly built render targets into their families (`init` step; `outputTexture` is already set). Also allocates the presented-LDR pool.
    func installTargets(_ t: Targets) throws {
        self.gbs.gbufferAlbedoMet    = t.albedoMet
        self.gbs.gbufferNormalRgh    = t.normalRgh
        self.gbs.gbufferEmission     = t.emission
        self.gbs.depthTexture        = t.depth
        self.gbs.previousDepthTexture = try Self.makeDepthLike(device: device, t.depth)
        self.gbs.hdrTexture          = t.hdr
        self.aos.aoTexture           = t.ao
        self.pfx.hdrCompositeTexture = t.hdrComposite
        self.rts.rtDiffuseTexture    = t.rtDiffuse
        self.pfx.velocityTexture     = t.velocity
        self.gbs.gbufferLayer        = t.layer
        self.pfx.historyA            = t.historyA
        self.pfx.historyB            = t.historyB
        self.pfx.taaResolvedTexture  = t.taaResolved
        self.pfx.bloomDownChain      = t.bloomDown
        self.pfx.bloomUpChain        = t.bloomUp
        self.pfx.halationBright = t.halationBright
        self.pfx.halationBlurH  = t.halationBlurH
        self.pfx.halationBlurV  = t.halationBlurV
        self.pfx.tonemapWriteTarget  = t.ldr
        // Phase 4.39 denoiser textures
        self.aos.aoFilteredTexture   = t.aoFiltered
        self.aos.aoHistoryA          = t.aoHistoryA
        self.aos.aoHistoryB          = t.aoHistoryB
        self.ssrs.ssrRawTexture       = t.ssrRaw
        self.ssrs.ssrHistoryA         = t.ssrHistoryA
        self.ssrs.ssrHistoryB         = t.ssrHistoryB
        self.rts.rtGIHistoryA        = t.rtGIHistoryA
        self.rts.rtGIHistoryB        = t.rtGIHistoryB
        self.aos.aoSampleCount       = t.aoSampleCount
        self.ssrs.ssrSampleCount      = t.ssrSampleCount
        self.rts.giSampleCount       = t.giSampleCount
        self.rts.giVariance          = t.giVariance
        self.rts.giVarianceB         = t.giVarianceB
        self.rts.giAtrousA           = t.giAtrousA
        self.rts.giAtrousB           = t.giAtrousB
        self.ssrs.sssDiffuseTexture   = t.sssDiffuse
        self.ssrs.sssBlurTmpTexture   = t.sssBlurTmp
        self.ssrs.sssBlurOutTexture   = t.sssBlurOut

        // Triple-buffer the presented LDR texture (see `ldrPool` docs above).
        // `t.ldr` becomes pool slot 0 (== outputTexture); allocate two more.
        var pool: [MTLTexture] = [t.ldr]
        for i in 1..<Self.ldrPoolCount {
            guard let extra = Self.makeLDRTexture(
                device: device, w: self.outputWidth, h: self.outputHeight,
                label: "Illuminatorama.ldr.\(i)") else {
                throw IlluminatoramaError.bufferAllocationFailed("ldrPool.\(i)")
            }
            pool.append(extra)
        }
        self.pfx.ldrPool = pool
        Self.clearLDRTextures(pool, on: engine.commandQueue)   // nothing unwritten may ever be presented — see `clearLDRTextures`
    }

    /// Per-frame buffer rings (instances, superquadric params, lights, frame uniforms) and the primitive meshes.
    func buildBuffersAndRings() throws {
        _ = IlluminatoramaInstance._assertStride240   // the Swift half of the layout guard (a lazy static never ran unreferenced)
        let initInstCap = 64
        let initLightCap = 16
        guard let iba = device.makeBuffer(
            length: MemoryLayout<IlluminatoramaInstance>.stride * initInstCap,
            options: .storageModeShared
        ) else { throw IlluminatoramaError.bufferAllocationFailed("instancesA") }
        guard let ibb = device.makeBuffer(
            length: MemoryLayout<IlluminatoramaInstance>.stride * initInstCap,
            options: .storageModeShared
        ) else { throw IlluminatoramaError.bufferAllocationFailed("instancesB") }
        iba.label = "Illuminatorama.instancesA"
        ibb.label = "Illuminatorama.instancesB"
        self.gbs.instanceBufferA = iba
        self.gbs.instanceBufferB = ibb
        self.gbs.instanceCapacity = initInstCap

        // Parallel superquadric-impostor param buffer (one entry per instance,
        // grouped-order-aligned with the instance buffer; see uploadInstances).
        // Ring-buffered (maxFramesInFlight) — params are read by the in-flight
        // frame's impostor draw, which can overlap the next frame's CPU write
        // when 2 frames pipeline. Motion vectors still come from current +
        // previous instance modelMatrix (the instance buffer's own ping-pong).
        let framesInFlight = IlluminatoramaRenderer.maxFramesInFlight
        let ringDevice = device   // local copy so the helper doesn't touch a still-initializing `self`
        func makeRing(_ length: Int, _ label: String, _ err: String) throws -> [MTLBuffer] {
            try (0..<framesInFlight).map { i in
                guard let b = ringDevice.makeBuffer(length: length, options: .storageModeShared)
                else { throw IlluminatoramaError.bufferAllocationFailed(err) }
                b.label = "\(label).\(i)"
                return b
            }
        }
        self.gbs.superquadricParamRing = try makeRing(
            MemoryLayout<IlluminatoramaInstance.SuperquadricParam>.stride * initInstCap,
            "Illuminatorama.superquadricParams", "superquadricParams")
        self.gbs.superquadricParamCapacity = initInstCap

        self.shd.pointLightRing = try makeRing(
            MemoryLayout<IlluminatoramaPointLight>.stride * initLightCap,
            "Illuminatorama.pointLights", "pointLights")
        self.shd.pointLightCapacity = initLightCap

        // Spot lights are bound at lighting kernel buffer(3). Same default
        // capacity as point lights — Eggs's rail-spotlight layout pushes
        // into the 20+ range so grow-on-demand will kick in early there.
        self.shd.spotLightRing = try makeRing(
            MemoryLayout<IlluminatoramaSpotLight>.stride * initLightCap,
            "Illuminatorama.spotLights", "spotLights")
        self.shd.spotLightCapacity = initLightCap
        // DH-0540 — the spot-profile placeholder (one float, 1.0): every kernel binds it; no spot reads it.
        guard let profilePlaceholder = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        else { throw IlluminatoramaError.bufferAllocationFailed("spotProfiles") }
        profilePlaceholder.contents().storeBytes(of: Float(1), as: Float.self)
        profilePlaceholder.label = "Illuminatorama.spotProfiles"
        self.shd.spotProfileBuffer = profilePlaceholder

        // Area lights are bound at lighting kernel buffer(4). Few per scene
        // (one softbox / window pane is typical); grow-on-demand covers more.
        self.lts.areaLightRing = try makeRing(
            MemoryLayout<IlluminatoramaAreaLight>.stride * initLightCap,
            "Illuminatorama.areaLights", "areaLights")
        self.lts.areaLightCapacity = initLightCap

        // Secondary directional fills are bound at lighting kernel buffer(5).
        // A SCN 3-point rig ships 1–2 (fill + back); grow-on-demand covers more.
        self.lts.extraDirectionalRing = try makeRing(
            MemoryLayout<IlluminatoramaDirectionalLight>.stride * initLightCap,
            "Illuminatorama.extraDirectionals", "extraDirectionals")
        self.lts.extraDirectionalCapacity = initLightCap

        // Interior daylight apertures (S3.5 Stage D) — lighting kernel buffer(7). One
        // per glazed opening; a whole storey is typically 10–30, grow-on-demand covers more.
        self.lts.interiorApertureRing = try makeRing(
            MemoryLayout<IlluminatoramaInteriorAperture>.stride * initLightCap,
            "Illuminatorama.interiorApertures", "interiorApertures")
        self.lts.interiorApertureCapacity = initLightCap

        self.lts.frameUniformRing = try makeRing(
            MemoryLayout<IlluminatoramaFrameUniforms>.stride,
            "Illuminatorama.frame", "frameUniforms")

        // Phase 1 ships three procedural primitives. Hosts can also call
        // `setMesh(_:_:)` to swap in their own.
        self.meshes[.box]    = IlluminatoramaMesh.unitBox(device: device)
        self.meshes[.sphere] = IlluminatoramaMesh.unitSphere(device: device)
        self.meshes[.ground] = IlluminatoramaMesh.unitGround(device: device)
    }

    /// Cover-blit pipeline (MTKView direct-present path).
    func buildCoverBlitPipeline(library: MTLLibrary) {
        // ── Cover-blit pipeline (for MTKView direct-present path) ────────
        if let vs = library.makeFunction(name: "illumi_coverblit_vs"),
           let fs = library.makeFunction(name: "illumi_coverblit_fs") {
            let bd = MTLRenderPipelineDescriptor()
            bd.label = "Illuminatorama.coverBlit"
            bd.vertexFunction = vs
            bd.fragmentFunction = fs
            bd.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
            self.pfx.coverBlitPipeline = try? device.makeRenderPipelineState(descriptor: bd)
        }
        if pfx.coverBlitPipeline == nil {
            Self.log.warning("illumi_coverblit shaders not found — present(to:) will no-op")
        }
    }
}
