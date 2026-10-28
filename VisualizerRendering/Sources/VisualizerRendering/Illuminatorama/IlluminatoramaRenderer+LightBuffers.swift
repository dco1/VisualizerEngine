import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - LightBufferState

/// Per-frame light and frame-uniform ring buffers.
/// Owned by the renderer as `lts`; every member keeps the name it had on the renderer.
@MainActor
final class LightBufferState {
    var areaLightRing: [MTLBuffer]!

    var areaLightCapacity: Int!
    var extraDirectionalRing: [MTLBuffer]!

    var interiorApertureRing: [MTLBuffer]!

    var interiorApertureCapacity: Int = 0
    var extraDirectionalCapacity: Int!
    var frameUniformRing: [MTLBuffer]!

    /// Which ring slot the CURRENT frame writes + binds. Advanced by exactly one
    /// per ACCEPTED `render()` (after the in-flight semaphore admits the frame),
    /// so two consecutive in-flight frames always use different slots. Sized to
    /// `maxFramesInFlight`; the GPU's max concurrency is bounded by the same
    /// semaphore, so a slot is never both written (CPU, frame N+1) and read
    /// (GPU, frame N) at once.
    var frameRingIndex: Int = 0
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    func uploadFrameUniforms() {
        advancePostFXEasing()
        let view = camera.viewMatrix
        // Apply Phase 2.7 sub-pixel projection jitter — translates the
        // projected clip-space x/y by a Halton(2,3) offset of up to
        // ±0.5 × taaJitterPixels pixels (in NDC units). The same matrix is
        // used everywhere (G-buffer rasterization, SSAO/SSR reconstruction,
        // lighting unprojection) so the internal consistency holds and TAA
        // averages out the sub-pixel offsets as supersampling.
        let unjitteredProj = camera.projectionMatrix
        var proj: simd_float4x4
        var jitterNDC: SIMD2<Float>
        if taaEnabled && taaJitterPixels > 0 {
            let h2 = Self.halton(pfx.taaFrameIndex &+ 1, base: 2) - 0.5
            let h3 = Self.halton(pfx.taaFrameIndex &+ 1, base: 3) - 0.5
            let jx = h2 * taaJitterPixels * 2.0 / Float(width)
            // NDC.y is up; pixel.y is down — but jitter is in NDC space here
            // and we want it expressed consistently with how the motion
            // vector flips Y, so leave +Y up here.
            let jy = h3 * taaJitterPixels * 2.0 / Float(height)
            var J = matrix_identity_float4x4
            J.columns.3 = SIMD4(jx, jy, 0, 1)
            proj = J * unjitteredProj
            jitterNDC = SIMD2(jx, jy)
        } else {
            proj = unjitteredProj
            jitterNDC = .zero
        }
        // Off-axis lens shift (opt-in; zero skips this block, so the matrices are untouched). The
        // same constant-NDC translation as the jitter, so it rides in `jitterNDC` too and stays out
        // of the motion vectors (`taaJitterDelta`).
        let shift = projectionShiftNDC
        if shift != .zero, shift.x.isFinite, shift.y.isFinite {
            var S = matrix_identity_float4x4
            S.columns.3 = SIMD4(shift.x, shift.y, 0, 1)
            proj = S * proj
            jitterNDC += shift
        }
        // `J` is a pure NDC translation (clip.xy += j · clip.w), so the jitter
        // enters every projected point as a constant additive NDC offset — which
        // is exactly why it can be removed from the velocity by subtraction
        // rather than by carrying a second pair of matrices. See
        // `IlluminatoramaFrameUniforms.taaJitterDelta`.
        pfx.lastFrameJitterNDC = jitterNDC
        let vp = proj * view
        gbs.gbufferCullVP = vp   // DH-0534: the G-buffer culls against exactly this
        let invVP = vp.inverse
        let invProj = proj.inverse
        let invView = view.inverse
        pfx.lastFrameViewProjection = vp
        var u = IlluminatoramaFrameUniforms(
            viewProjection: vp,
            view: view,
            projection: proj,
            invViewProjection: invVP,
            invProjection: invProj,
            invView: invView,
            cameraWorldPos: camera.position,
            directionalLightDir: simd_normalize(directionalLightDirection),
            directionalLightColor: directionalLightColor,
            ambientColor: ambientColor,
            exposure: pfx.easedExposure / preExposureDivisor,
            bloomThreshold: pfx.easedBloomThreshold * preExposureDivisor,
            bloomIntensity: pfx.easedBloomIntensity,
            pointLightCount: UInt32(pointLights.count),
            time: time,
            ssaoIntensity: ssaoIntensity,
            ssaoRadius: ssaoRadius,
            ssrIntensity: ssrIntensity,
            ssrMaxDistance: ssrMaxDistance,
            ssrThickness: ssrThickness,
            ssrMaxSteps: ssrMaxSteps,
            iblIntensity: iblIntensity,
            iblPrefilteredMipCount: UInt32(ibls.prefilteredMipViews.count),
            iblEnabled: iblEnabled ? 1 : 0,
            shadowVP0: shd.cascadeVPs[0],
            shadowVP1: shd.cascadeVPs[1],
            shadowVP2: shd.cascadeVPs[2],
            cascadeSplitsView: shd.cascadeSplitsView,
            shadowBias: shadowBias,
            shadowSlopeBias: shadowSlopeBias,
            shadowEnabled: shadowsEnabled ? 1 : 0,
            shadowPcfRadius: shadowPcfRadius,
            previousViewProjection: pfx.previousViewProjection,
            taaHistoryBlend: taaHistoryBlend,
            taaEnabled: taaEnabled ? 1 : 0,
            taaIsFirstFrame: pfx.taaNeedsFirstFrame ? 1 : 0,
            dfgLUTEnabled: dfgLUTEnabled ? 1 : 0,
            spotLightCount: UInt32(spotLights.count),
            spotShadowBias: spotShadowBias,
            tonemapSaturation: tonemapSaturation,
            iblDiffuseSaturation: iblDiffuseSaturation,
            autoExposureEnabled: autoExposureEnabled ? 1 : 0,
            autoExposureTargetEV: autoExposureTargetEV,
            autoExposureHalfLife: autoExposureHalfLife,
            debugTerm: debugTerm.rawValue,
            iblDiffuseDesaturation: iblDiffuseDesaturation,
            ssaoSpatialEnabled: ssaoSpatialEnabled ? 1 : 0,
            ssaoTemporalBlend: ssaoTemporalBlend,
            ssrTemporalEnabled: ssrTemporalEnabled ? 1 : 0,
            ssrTemporalBlend: ssrTemporalBlend,
            ssaoIsFirstFrame: aos.aoNeedsFirstFrame ? 1 : 0,
            ssrIsFirstFrame: ssrs.ssrNeedsFirstFrame ? 1 : 0,
            debandDitherEnabled: debandDitherEnabled ? (debandDitherExactSRGB ? 2 : 1) : 0,
            leafTransmission: max(0, leafTransmission),
            areaLightCount: UInt32(areaLights.count),
            areaLTCEnabled: (areaLTCOverride ?? ibls.ltcValidated) ? 1 : 0,
            directionalLightCount: UInt32(extraDirectionals.count),
            plushSheen: max(0, plushSheen),
            plushTransmission: max(0, plushTransmission)
        )
        // DH-0458 — lamp-shade fabric translucency. Written into the former `_padPhase3`
        // slot (default 0 in the memberwise init above), so a host that never sets it —
        // every Visualizer scene — ships a byte-identical uniform blob.
        u.shadeTransmission = max(0, shadeTransmission)
        // C1 — sun ownership, enforced here rather than left to the host. With
        // `rtSunOwnership == .rayTraced` the RT lighting pass computes the sun
        // (soft disc shadows), so the deferred directional + its cascades MUST go
        // dark for this frame or the sun lands twice. Same condition every
        // Visualizer host already applies by hand (`rtOn = rtEnabled &&
        // rtSupported`), so those scenes see no change — they zero the colour
        // themselves and resolve to `.automatic`-rayTraced anyway; this is the
        // belt for the next host that doesn't know the convention. `.automatic`
        // is deliberately NOT handled here: it MEANS "the host already did it".
        if rtSunOwnership == .rayTraced, rtEnabled, rtSupported, rtOpaqueLightingEnabled {
            u.directionalLightColor = .zero
            u.shadowEnabled = 0
        }
        // DH-0989 — `frameFlags` bit 1 (the Metal twin of this dead `shadowEnabled` slot): the
        // path-traced lane owns the diffuse indirect this frame, so the lighting pass hands ALL of
        // it over (`kFrameFlagPathOwnsIndirect`).
        if rtPathTracedGI && diffSkyHandoffActive { u.shadowEnabled |= 2 }
        // DH-1320 — the one-bounce lane's hand-off of the host's room-fill stand-ins (bits 16/32/64).
        if !rtPathTracedGI && rtGIReplacesDiffuseSky && diffSkyHandoffActive {
            u.shadowEnabled |= (rtGIOwnsRoomFill.rawValue & 7) << 4
        }
        // DH-1014 — the layered BSDF (`kFrameFlagLayeredMultiscatter` 4 / `…Sheen` 8).
        if dfgLUTEnabled && layeredMultiscatter { u.shadowEnabled |= 4 }
        if dfgLUTEnabled && layeredClothSheen { u.shadowEnabled |= 8 }
        if metalNoFlatAmbient { u.shadowEnabled |= 2048 }   // DH-1534 — `kFrameFlagMetalNoAmbient`
        if layeredDyedSheen { u.shadowEnabled |= 1024 }   // DH-1316 — `kFrameFlagDyedSheen` (bolted-on or layered sheen)
        if diffuseRoughnessEnabled { u.shadowEnabled |= 128 }   // DH-1301 — `kFrameFlagDiffuseRoughness`
        if layeredSpecularEnabled { u.shadowEnabled |= 256 }    // DH-1300 — `kFrameFlagLayeredSpecular`
        if dfgLUTEnabled && layeredCoatEnabled { u.shadowEnabled |= 512 }   // DH-1309 — `kFrameFlagLayeredCoat`
        if thinFilmEnabled { u.shadowEnabled |= 4096 }   // DH-1014 — `kFrameFlagThinFilm`
        if translucencyEnabled { u.shadowEnabled |= 8192 }   // DH-1302 — `kFrameFlagTranslucency`
        u.leafTransmittanceRatio = max(0, leafTransmittanceRatio)   // Signack S5
        // Tree wind (#58 #1): repurpose the two free pad floats as the vertex-
        // shader vegetation-wind knobs. _padPhase2A = strength (max canopy sway,
        // ~m), _padPhase2B = heading (radians). 0 strength → exact no-op (the
        // shader's applyTreeWind early-returns), so every other scene is unaffected.
        u._padPhase2A = treeWindStrength
        u._padPhase2B = treeWindHeading
        // Foliage-wind motion vector (DH-0492): the previous-frame wind is sampled at
        // `time − windPrevDelta` so a swaying canopy writes a real velocity TAA can track.
        // 0 (default) → previous sampled at the same time → exact no-op for every scene that
        // never sets it and every settled capture (clock frozen ⇒ delta 0).
        u.windPrevDelta = max(0, windPreviousDelta)
        // DH-0441 — the wide AO ring: a second GTAO march whose falloff is the near
        // kernel's complement, weighing only occluders beyond `ssaoRadius` out to
        // `ssaoFarRadius`. `ssaoFarIntensity` 0 (default) ⇒ `illumi_ssao` never marches the
        // ring and the AO field is byte-identical. See IlluminatoramaSSAO.metal.
        u.ssaoFarRadius = ssaoFarRadius
        u.ssaoFarIntensity = ssaoFarIntensity
        // Chromatic aberration: tonemap CA strength (repurposes the former
        // _padPlush0 slot). 0 → exact no-op (the tonemap branch is gated on it).
        // Eased value (see advancePostFXEasing) so slider drags glide.
        u.chromaticAberration = max(0, pfx.easedChromaticAberration)
        // Spherical aberration: radial blur, 0 → exact no-op.
        u.sphericalAberration = max(0, pfx.easedSphericalAberration)
        // Axial chromatic aberration ("purple fringing"): strength + dark-side
        // tint. 0 strength → exact no-op (the tonemap branch is gated on it).
        u.fringe = max(0, pfx.easedFringe)
        u.fringeTintR = pfx.easedFringeTint.x
        u.fringeTintG = pfx.easedFringeTint.y
        u.fringeTintB = pfx.easedFringeTint.z
        // Vignette + film grain (issue #65). Applied directly (no easing) — 0 → no-op.
        u.vignetteStrength  = max(0, vignetteStrength)
        u.vignetteExtent    = max(0, min(1, vignetteExtent))
        u.filmGrainStrength = max(0, filmGrainStrength)
        u.filmGrainSize     = max(1, filmGrainSize)
        // Colour-grade LUT (issue #65). 0 → no-op; size feeds the half-texel inset.
        u.colorLUTAmount    = max(0, min(1, colorLUTAmount))
        u.colorLUTSize      = Float(pfx.colorLUTSize)
        // Velocity-buffer motion blur (issue #65). 0 → no-op. The env override
        // (VIZ_ILLUMI_MOTIONBLUR) wins over the panel value for headless verify.
        let effMotionBlur = Self.motionBlurEnvOverride ?? motionBlurStrength
        u.motionBlurStrength = Self.mvDebugOverlay ? -1.0 : max(0, effMotionBlur)
        u.motionBlurMaxPx    = max(0, motionBlurMaxPx)
        // Screen-space contact shadows (issue #65). 0 strength → no-op (the
        // lighting kernel skips the march). Length/thickness are world units.
        // The env override (VIZ_ILLUMI_CONTACTSHADOW) wins over the panel value
        // for headless A/B verify.
        let effContactShadow = Self.contactShadowEnvOverride ?? contactShadowStrength
        u.contactShadowStrength  = max(0, min(1, effContactShadow))
        u.contactShadowLength    = max(0, contactShadowLength)
        u.contactShadowSteps     = min(max(contactShadowSteps, 1), 32)
        u.contactShadowThickness = max(0, contactShadowThickness)
        // Aerial perspective. 0 → exact no-op (the lighting kernel skips the blend).
        u.aerialPerspectiveDensity = max(0, aerialPerspectiveDensity)
        let apScale = max(1e-4, aerialPerspectiveAirlightScale)
        u.aerialAirlightScale = aerialPerspectiveAirlightFromHorizon ? -apScale
            : (abs(aerialPerspectiveAirlightScale - 1) < 1e-6 ? 0 : apScale)
        // Screen-space subsurface scattering (issue #65). 0 strength → no-op (the
        // lighting pass skips the side-buffer write; the blur/composite passes
        // aren't encoded). The env override (VIZ_ILLUMI_SSS) wins for headless verify.
        let effSSS = Self.sssStrengthEnvOverride ?? sssStrength
        u.sssStrength = max(0, min(1, effSSS))
        u.sssRadius   = max(0, Self.sssRadiusEnvOverride ?? sssRadius)
        u.sssTintR    = max(0, sssTint.x)
        u.sssTintG    = max(0, sssTint.y)
        u.sssTintB    = max(0, sssTint.z)
        u.sssDebugForceAll = Self.sssForceAllEnv ? 1.0 : 0.0
        // Film halation: intensity / threshold / halo radius (fraction of frame height)
        // + tail tint. Intensity 0 (the default) → the tonemap branch is skipped AND
        // `encodeHalationPasses` doesn't dispatch, so a non-opting scene is byte-identical.
        u.halationParams = SIMD4(max(0, pfx.easedHalationIntensity),
                                 pfx.easedHalationThreshold * preExposureDivisor,
                                 max(0, pfx.easedHalationRadius),
                                 0)
        u.halationTint = SIMD4(pfx.easedHalationTint.x, pfx.easedHalationTint.y, pfx.easedHalationTint.z, 0)
        // Bloom pyramid (S1.2): soft knee (fraction of the threshold) / upsample
        // scatter / tent radius. Scatter is clamped to [0,1] because the up-chain
        // blend must stay CONVEX — outside that range the geometric series stops
        // summing to 1 and the chain gains or loses energy.
        u.sensorNoise = SIMD4(max(0, sensorNoiseStrength), max(1, sensorElectronsPerUnit),
                              max(0, sensorReadNoise), min(max(sensorChannelCorrelation, 0), 1))
        u.filmMedium = SIMD4(max(0, filmDensityGrain), min(max(filmDensityGrainCorrelation, 0), 1), aoMultiBounce ? 1 : 0, 0)
        u.bloomParams = SIMD4(max(0, bloomSoftKnee),
                              min(max(bloomScatter, 0), 1),
                              max(0, bloomTentRadius),
                              0)
        // DH-1001 — physical glare: refit (only when the lens / resolution / observer changes) and
        // hand the shaders η. The pixel angle is derived exactly like `nightPixAngle` (tan(fovY/2) =
        // 1/P[1][1]) over the PYRAMID's mip0 height, the grid the measured kernels are in.
        if bloomPhysicalGlare, !camera.isOrthographic, let mip0 = pfx.bloomDownChain.first, pfx.bloomDownChain.count >= 2 {
            let texelRad = (2.0 / max(camera.projectionMatrix[1][1], 1e-6)) / Float(max(mip0.height, 1))
            let fitLevels = Self.bloomGlareFitLevelCount(halfW: mip0.width, halfH: mip0.height)
            if lastBloomGlareFit == nil || pfx.glareFitKey?.texelRad != texelRad || pfx.glareFitKey?.levels != fitLevels
                || pfx.glareFitKey?.model != bloomGlareModel {
                lastBloomGlareFit = IlluminatoramaGlareSpread.fit(
                    texelRadians: Double(texelRad), levels: fitLevels, model: bloomGlareModel)
                pfx.glareFitKey = (texelRad, fitLevels, bloomGlareModel)
            }
            // DH-0998 — the far-field tail's view geometry: the half field of view's tangents.
            pfx.glareTailTanHalf = SIMD2(1 / max(camera.projectionMatrix[0][0], 1e-6),
                                     1 / max(camera.projectionMatrix[1][1], 1e-6))
            let eta = Float(lastBloomGlareFit?.energy ?? 0)
            u.bloomIntensity = eta
            u.bloomParams = SIMD4(0, 0, 1, eta)
        } else {
            lastBloomGlareFit = nil
        }
        // Diagram look. `mix` 0 (the default) keeps the tonemap branch unentered, so
        // this is an exact no-op for every scene that never opts in. The `w` slot is
        // the ORTHOGRAPHIC flag, not a look control: it is read straight off the live
        // camera so the shaders' view-vector derivations can never disagree with the
        // projection actually being used — a hand-set flag here would be a second
        // source of truth for "is the eye at infinity".
        u.diagramParams = SIMD4(min(max(diagramMix, 0), 1),
                                min(max(diagramWrap, 0), 1),
                                min(max(diagramOutlineStrength, 0), 1),
                                camera.isOrthographic ? 1 : 0)
        u.diagramOutline = SIMD4(diagramOutlineColor.x, diagramOutlineColor.y,
                                 diagramOutlineColor.z, 0)
        u.diagramEdge = SIMD4(max(0, diagramEdgeDepthSensitivity),
                              max(0, diagramEdgeNormalSensitivity),
                              max(0.25, diagramEdgeThickness),
                              0)
        // S4.2 — the jitter the velocity buffer must NOT carry. Both matrices it
        // differences are jittered (this frame's, and last frame's inside
        // `previousViewProjection`), so the delta of the two offsets is what is
        // left over on top of the surface's real motion. Zero whenever jitter is
        // off, which is every frame of every scene that never opts in.
        // zw — DH-1365: the reconstruction filter (kernel, radius); (0, 0) ⇒ the legacy 4-tap.
        u.taaJitterDelta = SIMD4(jitterNDC.x - pfx.previousJitterNDC.x,
                                 jitterNDC.y - pfx.previousJitterNDC.y,
                                 reconstructionFilter.uniform.x, reconstructionFilter.uniform.y)
        // Phase 9 — film LUT strength: 0 when no LUT is bound (bypasses the shader branch).
        u.filmLUTStrength = filmLUTTexture != nil ? max(0, min(1, filmLUTStrength)) : 0
        u.filmLUTSize = max(2, filmLUTSize)
        u.naturalVignetteK = max(0, naturalVignetteK)
        u.displayTransform = displayTransform.rawValue
        // Tonemap colour-grade. Neutral defaults (6500/0/1/1/1) are exact no-ops in
        // the shader, so scenes that never touch these are byte-for-byte unchanged.
        u.whiteBalanceK = whiteBalanceK
        u.tint = tint
        u.shadows = shadows
        u.highlights = highlights
        u.contrast = contrast
        // Phase 7 — opt-in anti-tiling. Clamp to [0,1]; 0 (default) makes the shader
        // take the plain single-sample path, so opted-out scenes are unchanged.
        u.antiTilingStrength = max(0, min(1, antiTilingStrength))
        // Point-light cube shadow bias (0 when the feature is off ⇒ no behaviour change).
        u.pointShadowBias = pointShadowsEnabled ? pointShadowBias : 0
        // Scotopic (Purkinje) night desaturation. 0 (default) ⇒ the tonemap branch
        // never runs ⇒ byte-identical; hosts fade it in with their night blend.
        u.scotopicDesaturation = max(0, scotopicDesaturation)
        // Photographic finish — highlight chroma roll-off + split tone. 0 / 6500 /
        // 6500 (the defaults) make every branch an exact no-op, so Visualizer and any
        // scene that never opts in is byte-identical.
        u.highlightChromaRolloff = max(0, min(1, highlightChromaRolloff))
        u.shadowTemperatureK = max(1000, min(20000, shadowTemperatureK))
        u.highlightTemperatureK = max(1000, min(20000, highlightTemperatureK))
        // DH-0715 — live-lane look-match. Strength 0 (default) ⇒ byte-identical.
        u.liveLookMatchStrength = max(0, min(1, liveLookMatchStrength))
        u.liveLookGIDarken = liveLookGIDarken
        u.liveLookGIWarmth = max(0, min(4, liveLookGIWarmth))
        u.liveLookAODarken = effectiveLiveLookAODarken   // DH-0785 — never on top of traced AO
        u.tagGlowGain = max(0, tagGlowGain)
        u.fieldSink = min(max(fieldSinkProgress, 0), 1)
        u.prevFieldSink = vols.lastFieldSinkProgress
        vols.lastFieldSinkProgress = u.fieldSink
        // Interior day-light separation. Mask 0 (default) ⇒ the kernel's factors stay
        // exactly 1.0 ⇒ byte-identical for every scene that never opts in.
        u.interiorMask = interiorLayerMask
        u.interiorIBLUp = max(0, interiorIBLUp)
        u.interiorIBLSide = max(0, interiorIBLSide)
        u.interiorAmbient = max(0, interiorAmbient)
        // Interior irradiance bands. Weight 0 (default) ⇒ the diffuse path is the
        // exact cube sample ⇒ byte-identical for every scene that never opts in.
        let irrW = max(0, min(1, interiorIrradianceWeight))
        u.interiorIrrUp = SIMD4(simd_max(interiorIrradianceUp, .zero), irrW)
        u.setInteriorRoomGains(interiorRoomGains, enabled: interiorRoomGainsEnabled)
        // The two spare .w lanes carry the aperture GRADATION (S3.5 Stage D):
        // side.w = strength (0 = off), down.w = aperture count (float-encoded; exact
        // for any realistic count). Both 0 by default ⇒ the kernel's factor is 1.0.
        u.interiorIrrSide = SIMD4(simd_max(interiorIrradianceSide, .zero),
                                  max(0, min(1, interiorApertureGradation)))
        u.interiorIrrDown = SIMD4(simd_max(interiorIrradianceDown, .zero),
                                  Float(interiorApertures.count))
        // S4.1 — ray-traced soft sun shadows. Read ONLY by the kLightingRTSunShadow
        // pipeline variant; for every other variant these are dead uniforms, so the
        // writes cannot move a non-opting scene. The seed WALKS only while TAA is
        // accumulating and FREEZES at 0 otherwise — the same contract as the glass
        // pass's `frameSeed` below: with no accumulator, a walking seed turns the
        // cone-sampling noise into speckle that crawls on a perfectly static frame.
        u.rtSunShadowSeed = taaEnabled ? pfx.taaFrameIndex : 0
        u.rtSunShadowAngle = max(0.0005, rtSunSoftnessRad)
        // DH-0856 — clamp raised 8 -> 32: at the photo lane's fixed 32-frame settle budget,
        // 8 cone-sampled rays/frame left the traced penumbra's Monte-Carlo noise (edge
        // residual std ~5.2px on the "Primary Bedroom, Facing North" repro) far short of the
        // cascade-map default's analytic PCF (~1.6px) — measured directly, not assumed. Raising
        // the cap costs real per-frame GPU time (linear in ray count) but nothing else; the
        // shader-side clamp in `rtSunSoftVisibility` (IlluminatoramaLighting.metal) moved with
        // it so a value above 8 is not silently re-clamped one level down.
        u.rtSunShadowRayCount = UInt32(max(1, min(32, rtShadowRays)))
        u.rtAreaShadowRayCount = rtSunSoftShadowsEnabled ? UInt32(max(0, min(8, rtAreaShadowRays))) : 0
        // Analytic night sky. All-zero defaults keep the kernel's sky branch an
        // exact no-op; hosts fade the brightnesses with nightBlend themselves.
        u.nightSkyParams = SIMD4(max(0, nightSkyStarBrightness),
                                 max(0, nightSkyMoonIntensity),
                                 max(0, nightSkyMoonAngularRadius), 0)
        let moonDirSafe = nightSkyMoonDirection == .zero
            ? SIMD3<Float>(0, 1, 0) : simd_normalize(nightSkyMoonDirection)
        let sunDirSafe = nightSkySunDirection == .zero
            ? SIMD3<Float>(0, -1, 0) : simd_normalize(nightSkySunDirection)
        u.nightMoonDir = SIMD4(moonDirSafe, 0)
        u.nightSunDir = SIMD4(sunDirSafe, 0)
        (u.nightSkyExtra, u.nightSkyExtra2, u.nightCelestial) = physicalNightSkyClusters()
        u.scotopicParams = SIMD4<Float>(max(0, scotopicKnee), max(0, scotopicTint.x), max(0, scotopicTint.y), max(0, scotopicTint.z))
        // Pre-exposure + hue-stable toe (all zero by default ⇒ the shader's legacy paths).
        let toeHi = max(0, hueStableToe.y)
        u.rtSurfaceSnap = SIMD4<Float>(effectiveRTSurfaceSnapULPs, 0, 0, 0)
        u.displayParams = SIMD4<Float>(preExposureDivisor == 1 ? 0 : preExposureDivisor,
                                       toeHi > 0 ? max(0, min(hueStableToe.x, toeHi * 0.999)) : 0, toeHi,
                                       toeHi > 0 && hueStableToeBareLevel ? 1 : 0)
        // Defocus shading rate's gate (read only by the coarse lighting variants; 0 ⇒ off).
        u.lightingCoarseMinCoC = lightingShadingRateRequested ? max(4, lightingDefocusShadingMinCoC) : 0
        // Lens flare: project the primary sun's direction to screen uv. w carries the
        // on-screen weight — a smooth fade as the sun leaves the frame, hard 0 behind
        // the camera (clip.w ≤ 0). Strength 0 (default) leaves the whole cluster zero,
        // an exact shader no-op.
        u.lensFlareParams = .zero
        if lensFlareIntensity > 0 {
            let toSun = simd_normalize(u.directionalLightDir)
            let clip = u.viewProjection * SIMD4<Float>(u.cameraWorldPos + toSun * 500, 1)
            if clip.w > 0 {
                let ndc = SIMD2<Float>(clip.x / clip.w, clip.y / clip.w)
                let uv = SIMD2<Float>(ndc.x * 0.5 + 0.5, 1 - (ndc.y * 0.5 + 0.5))
                let edge = max(abs(ndc.x), abs(ndc.y))
                let weight = max(0, min(1, (1.25 - edge) / 0.35))
                u.lensFlareParams = SIMD4(max(0, lensFlareIntensity), uv.x, uv.y, weight)
            }
        }
        lastLensFlareParams = u.lensFlareParams
        // DH-0140 — decide the G-buffer lane for THIS frame here, before any pass encodes, so the
        // pass that writes the sixth target and the kernel that reads it agree by construction.
        extendedGBufferActive = extendedGBufferEnabled && ensureExtendedGBufferResources()
        if !extendedGBufferEnabled, gbs.gbufferMaterial != nil { gbs.gbufferMaterial = nil }   // the lane left: give the memory back
        u.extendedGBuffer = extendedGBufferActive ? 1 : 0
        memcpy(frameUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaFrameUniforms>.stride)
    }

    func uploadPointLights() {
        guard !pointLights.isEmpty else { return }
        pointLights.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                pointLightBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaPointLight>.stride * pointLights.count
            )
        }
    }

    func uploadSpotLights() {
        guard !spotLights.isEmpty else { return }
        spotLights.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                spotLightBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaSpotLight>.stride * spotLights.count
            )
        }
        // DH-0540 — a `profileIndex` past the table the host actually supplied is the analytic cone, never an
        // out-of-bounds read; clamped in the COPY so the host's array keeps what it authored.
        let rows = spotLightProfileRowCount
        let dst = spotLightBuffer.contents().bindMemory(to: IlluminatoramaSpotLight.self, capacity: spotLights.count)
        for i in 0..<spotLights.count where dst[i].profileIndex >= Int32(rows) { dst[i].profileIndex = -1 }
    }

    /// Daydream DH-0540 — hand the renderer the spot-profile table: one row per curated luminaire + hemisphere,
    /// each exactly `illuminatoramaSpotProfileSampleCount` floats (intensity multiplier over 0..90° off the cone
    /// axis, flux-matched by the host). A spot names its row with `IlluminatoramaSpotLight.profileIndex`; −1 or a
    /// row past this table keeps the analytic cone. Allocates a fresh buffer each call.
    public func setSpotLightProfiles(_ rows: [[Float]]) {
        for r in rows {
            precondition(r.count == illuminatoramaSpotProfileSampleCount,
                         "a spot-profile row has \(illuminatoramaSpotProfileSampleCount) samples, got \(r.count)")
        }
        let flat = rows.flatMap { $0 }
        let length = max(1, flat.count) * MemoryLayout<Float>.stride
        guard let buf = device.makeBuffer(length: length, options: .storageModeShared) else { return }
        if flat.isEmpty { buf.contents().storeBytes(of: Float(1), as: Float.self) }
        else { flat.withUnsafeBufferPointer { memcpy(buf.contents(), $0.baseAddress!, flat.count * MemoryLayout<Float>.stride) } }
        buf.label = "Illuminatorama.spotProfiles"
        shd.spotProfileBuffer = buf
        spotLightProfileRowCount = rows.count
    }

    func uploadInteriorApertures() {
        guard !interiorApertures.isEmpty else { return }
        interiorApertures.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                interiorApertureBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaInteriorAperture>.stride * interiorApertures.count
            )
        }
    }

    func uploadAreaLights() {
        guard !areaLights.isEmpty else { return }
        areaLights.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                areaLightBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaAreaLight>.stride * areaLights.count
            )
        }
    }

    func uploadExtraDirectionals() {
        guard !extraDirectionals.isEmpty else { return }
        extraDirectionals.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            memcpy(
                extraDirectionalBuffer.contents(),
                base,
                MemoryLayout<IlluminatoramaDirectionalLight>.stride * extraDirectionals.count
            )
        }
    }
}
