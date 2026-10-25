import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `SkyKnobs` — storage lives in IlluminatoramaSkyKnobs.swift.
extension IlluminatoramaRenderer {
    public var treeWindStrength: Float { get { kSky.treeWindStrength } set { kSky.treeWindStrength = newValue } _modify { yield &kSky.treeWindStrength } }
    public var treeWindHeading: Float { get { kSky.treeWindHeading } set { kSky.treeWindHeading = newValue } _modify { yield &kSky.treeWindHeading } }
    public var windPreviousDelta: Float { get { kSky.windPreviousDelta } set { kSky.windPreviousDelta = newValue } _modify { yield &kSky.windPreviousDelta } }
    public var volumetricEnabled: Bool { get { kSky.volumetricEnabled } set { kSky.volumetricEnabled = newValue } _modify { yield &kSky.volumetricEnabled } }
    public var volFogDensity: Float { get { kSky.volFogDensity } set { kSky.volFogDensity = newValue } _modify { yield &kSky.volFogDensity } }
    public var volScatterStrength: Float { get { kSky.volScatterStrength } set { kSky.volScatterStrength = newValue } _modify { yield &kSky.volScatterStrength } }
    public var volAnisotropy: Float { get { kSky.volAnisotropy } set { kSky.volAnisotropy = newValue } _modify { yield &kSky.volAnisotropy } }
    public var volSteps: Int { get { kSky.volSteps } set { kSky.volSteps = newValue } _modify { yield &kSky.volSteps } }
    public var volMaxDistance: Float { get { kSky.volMaxDistance } set { kSky.volMaxDistance = newValue } _modify { yield &kSky.volMaxDistance } }
    public var volWindowX: Float { get { kSky.volWindowX } set { kSky.volWindowX = newValue } _modify { yield &kSky.volWindowX } }
    public var volWindowRect: SIMD4<Float> { get { kSky.volWindowRect } set { kSky.volWindowRect = newValue } _modify { yield &kSky.volWindowRect } }
    public var volFeather: Float { get { kSky.volFeather } set { kSky.volFeather = newValue } _modify { yield &kSky.volFeather } }
    public var volOutdoorMode: Bool { get { kSky.volOutdoorMode } set { kSky.volOutdoorMode = newValue } _modify { yield &kSky.volOutdoorMode } }
    public var volumetricRTEnabled: Bool { get { kSky.volumetricRTEnabled } set { kSky.volumetricRTEnabled = newValue } _modify { yield &kSky.volumetricRTEnabled } }
    public var volRTDensity: Float { get { kSky.volRTDensity } set { kSky.volRTDensity = newValue } _modify { yield &kSky.volRTDensity } }
    public var volRTAnisotropy: Float { get { kSky.volRTAnisotropy } set { kSky.volRTAnisotropy = newValue } _modify { yield &kSky.volRTAnisotropy } }
    public var volRTSteps: Int { get { kSky.volRTSteps } set { kSky.volRTSteps = newValue } _modify { yield &kSky.volRTSteps } }
    public var volRTPixelStride: Int { get { kSky.volRTPixelStride } set { kSky.volRTPixelStride = newValue } _modify { yield &kSky.volRTPixelStride } }
    public var volRTMaxDistance: Float { get { kSky.volRTMaxDistance } set { kSky.volRTMaxDistance = newValue } _modify { yield &kSky.volRTMaxDistance } }
    public var volRTRoofReach: Float { get { kSky.volRTRoofReach } set { kSky.volRTRoofReach = newValue } _modify { yield &kSky.volRTRoofReach } }
    public var volRTSunScatter: Float { get { kSky.volRTSunScatter } set { kSky.volRTSunScatter = newValue } _modify { yield &kSky.volRTSunScatter } }
    public var volRTSkyScatter: Float { get { kSky.volRTSkyScatter } set { kSky.volRTSkyScatter = newValue } _modify { yield &kSky.volRTSkyScatter } }
    public var volRTDustDensity: Float { get { kSky.volRTDustDensity } set { kSky.volRTDustDensity = newValue } _modify { yield &kSky.volRTDustDensity } }
    public var volRTDustAnisotropy: Float { get { kSky.volRTDustAnisotropy } set { kSky.volRTDustAnisotropy = newValue } _modify { yield &kSky.volRTDustAnisotropy } }
    public var volRTDustAdmittanceRef: Float { get { kSky.volRTDustAdmittanceRef } set { kSky.volRTDustAdmittanceRef = newValue } _modify { yield &kSky.volRTDustAdmittanceRef } }
    public var volRTAdmittanceGrid: Int { get { kSky.volRTAdmittanceGrid } set { kSky.volRTAdmittanceGrid = newValue } _modify { yield &kSky.volRTAdmittanceGrid } }
    public var volSpotBeamsEnabled: Bool { get { kSky.volSpotBeamsEnabled } set { kSky.volSpotBeamsEnabled = newValue } _modify { yield &kSky.volSpotBeamsEnabled } }
    public var volSpotBeamDensity: Float { get { kSky.volSpotBeamDensity } set { kSky.volSpotBeamDensity = newValue } _modify { yield &kSky.volSpotBeamDensity } }
    public var volSpotBeamStrength: Float { get { kSky.volSpotBeamStrength } set { kSky.volSpotBeamStrength = newValue } _modify { yield &kSky.volSpotBeamStrength } }
    public var volSpotBeamAnisotropy: Float { get { kSky.volSpotBeamAnisotropy } set { kSky.volSpotBeamAnisotropy = newValue } _modify { yield &kSky.volSpotBeamAnisotropy } }
    public var volSpotBeamSteps: Int { get { kSky.volSpotBeamSteps } set { kSky.volSpotBeamSteps = newValue } _modify { yield &kSky.volSpotBeamSteps } }
    public var volSpotBeamMaxDistance: Float { get { kSky.volSpotBeamMaxDistance } set { kSky.volSpotBeamMaxDistance = newValue } _modify { yield &kSky.volSpotBeamMaxDistance } }
    public var volSpotBeamShadows: Bool { get { kSky.volSpotBeamShadows } set { kSky.volSpotBeamShadows = newValue } _modify { yield &kSky.volSpotBeamShadows } }
    public var volSpotBeamSpotLimit: Int { get { kSky.volSpotBeamSpotLimit } set { kSky.volSpotBeamSpotLimit = newValue } _modify { yield &kSky.volSpotBeamSpotLimit } }
    public var volEmissiveColumn: EmissiveColumn? { get { kSky.volEmissiveColumn } set { kSky.volEmissiveColumn = newValue } _modify { yield &kSky.volEmissiveColumn } }
    public var nightSkyStarBrightness: Float { get { kSky.nightSkyStarBrightness } set { kSky.nightSkyStarBrightness = newValue } _modify { yield &kSky.nightSkyStarBrightness } }
    public var nightSkyMoonIntensity: Float { get { kSky.nightSkyMoonIntensity } set { kSky.nightSkyMoonIntensity = newValue } _modify { yield &kSky.nightSkyMoonIntensity } }
    public var nightSkyMoonDirection: SIMD3<Float> { get { kSky.nightSkyMoonDirection } set { kSky.nightSkyMoonDirection = newValue } _modify { yield &kSky.nightSkyMoonDirection } }
    public var nightSkySunDirection: SIMD3<Float> { get { kSky.nightSkySunDirection } set { kSky.nightSkySunDirection = newValue } _modify { yield &kSky.nightSkySunDirection } }
    public var nightSkyMoonAngularRadius: Float { get { kSky.nightSkyMoonAngularRadius } set { kSky.nightSkyMoonAngularRadius = newValue } _modify { yield &kSky.nightSkyMoonAngularRadius } }
    public var nightSkyModel: VolumetricCloudRenderer.Params.NightSkyModel { get { kSky.nightSkyModel } set { kSky.nightSkyModel = newValue } _modify { yield &kSky.nightSkyModel } }
    public var nightSkyMoonHalo: Float { get { kSky.nightSkyMoonHalo } set { kSky.nightSkyMoonHalo = newValue } _modify { yield &kSky.nightSkyMoonHalo } }
    public var nightSkyEarthshine: Float { get { kSky.nightSkyEarthshine } set { kSky.nightSkyEarthshine = newValue } _modify { yield &kSky.nightSkyEarthshine } }
    public var nightSkyMilkyWay: Float { get { kSky.nightSkyMilkyWay } set { kSky.nightSkyMilkyWay = newValue } _modify { yield &kSky.nightSkyMilkyWay } }
    public var nightSkyTwinkle: Float { get { kSky.nightSkyTwinkle } set { kSky.nightSkyTwinkle = newValue } _modify { yield &kSky.nightSkyTwinkle } }
    public var nightSkyRadiance: Float { get { kSky.nightSkyRadiance } set { kSky.nightSkyRadiance = newValue } _modify { yield &kSky.nightSkyRadiance } }
    public var nightSkyCelestialOrientation: SIMD4<Float> { get { kSky.nightSkyCelestialOrientation } set { kSky.nightSkyCelestialOrientation = newValue } _modify { yield &kSky.nightSkyCelestialOrientation } }
    public var nightSkyStarLimitingMagnitude: Float { get { kSky.nightSkyStarLimitingMagnitude } set { kSky.nightSkyStarLimitingMagnitude = newValue } _modify { yield &kSky.nightSkyStarLimitingMagnitude } }
    public var inViewCloudsEnabled: Bool { get { kSky.inViewCloudsEnabled } set { kSky.inViewCloudsEnabled = newValue } _modify { yield &kSky.inViewCloudsEnabled } }
    public var cumulusInViewEnabled: Bool { get { kSky.cumulusInViewEnabled } set { kSky.cumulusInViewEnabled = newValue } _modify { yield &kSky.cumulusInViewEnabled } }
    public var cumulusInViewScale: Int { get { kSky.cumulusInViewScale } set { kSky.cumulusInViewScale = newValue } _modify { yield &kSky.cumulusInViewScale } }
    public var cumulusInViewInterleave: Int { get { kSky.cumulusInViewInterleave } set { kSky.cumulusInViewInterleave = newValue } _modify { yield &kSky.cumulusInViewInterleave } }
    public var cumulusInViewMaxSamples: Int { get { kSky.cumulusInViewMaxSamples } set { kSky.cumulusInViewMaxSamples = newValue } _modify { yield &kSky.cumulusInViewMaxSamples } }
    public var cumulusInViewSteps: Int { get { kSky.cumulusInViewSteps } set { kSky.cumulusInViewSteps = newValue } _modify { yield &kSky.cumulusInViewSteps } }
    public var cumulusInViewLightSteps: Int { get { kSky.cumulusInViewLightSteps } set { kSky.cumulusInViewLightSteps = newValue } _modify { yield &kSky.cumulusInViewLightSteps } }
    public var cumulusInViewDetail: Float { get { kSky.cumulusInViewDetail } set { kSky.cumulusInViewDetail = newValue } _modify { yield &kSky.cumulusInViewDetail } }
    public var cumulusInViewReprojection: Bool { get { kSky.cumulusInViewReprojection } set { kSky.cumulusInViewReprojection = newValue } _modify { yield &kSky.cumulusInViewReprojection } }
    public var cumulusInViewFadeFrames: Int { get { kSky.cumulusInViewFadeFrames } set { kSky.cumulusInViewFadeFrames = newValue } _modify { yield &kSky.cumulusInViewFadeFrames } }
    public var cloudShadowEnabled: Bool { get { kSky.cloudShadowEnabled } set { kSky.cloudShadowEnabled = newValue } _modify { yield &kSky.cloudShadowEnabled } }
    public var cloudShadowResolution: Int { get { kSky.cloudShadowResolution } set { kSky.cloudShadowResolution = newValue } _modify { yield &kSky.cloudShadowResolution } }
    public var cloudShadowExtent: Float { get { kSky.cloudShadowExtent } set { kSky.cloudShadowExtent = newValue } _modify { yield &kSky.cloudShadowExtent } }
    public var cloudShadowSteps: Int { get { kSky.cloudShadowSteps } set { kSky.cloudShadowSteps = newValue } _modify { yield &kSky.cloudShadowSteps } }
    public var cloudShadowGroundY: Float { get { kSky.cloudShadowGroundY } set { kSky.cloudShadowGroundY = newValue } _modify { yield &kSky.cloudShadowGroundY } }
    public var cloudShadowStrength: Float { get { kSky.cloudShadowStrength } set { kSky.cloudShadowStrength = newValue } _modify { yield &kSky.cloudShadowStrength } }
    public var cloudShadowHeightLimit: Float { get { kSky.cloudShadowHeightLimit } set { kSky.cloudShadowHeightLimit = newValue } _modify { yield &kSky.cloudShadowHeightLimit } }
    public var inViewSkyTime: Float? { get { kSky.inViewSkyTime } set { kSky.inViewSkyTime = newValue } _modify { yield &kSky.inViewSkyTime } }
    public var inViewLavaFade: Float? { get { kSky.inViewLavaFade } set { kSky.inViewLavaFade = newValue } _modify { yield &kSky.inViewLavaFade } }
    public var inViewCirrusTime: Float? { get { kSky.inViewCirrusTime } set { kSky.inViewCirrusTime = newValue } _modify { yield &kSky.inViewCirrusTime } }
    public var inViewCirrusThreads: Float? { get { kSky.inViewCirrusThreads } set { kSky.inViewCirrusThreads = newValue } _modify { yield &kSky.inViewCirrusThreads } }
    public var tagGlowGain: Float { get { kSky.tagGlowGain } set { kSky.tagGlowGain = newValue } _modify { yield &kSky.tagGlowGain } }
    public var fieldSinkProgress: Float { get { kSky.fieldSinkProgress } set { kSky.fieldSinkProgress = newValue } _modify { yield &kSky.fieldSinkProgress } }
    public var inViewCloudsResolutionScale: Float { get { kSky.inViewCloudsResolutionScale } set { kSky.inViewCloudsResolutionScale = newValue } _modify { yield &kSky.inViewCloudsResolutionScale } }
    public var cloudNoiseTexture: MTLTexture? { get { kSky.cloudNoiseTexture } set { kSky.cloudNoiseTexture = newValue } _modify { yield &kSky.cloudNoiseTexture } }
    public var cloudSkyUniforms: MTLBuffer? { get { kSky.cloudSkyUniforms } set { kSky.cloudSkyUniforms = newValue } _modify { yield &kSky.cloudSkyUniforms } }
    public var cloudBurstLights: MTLBuffer? { get { kSky.cloudBurstLights } set { kSky.cloudBurstLights = newValue } _modify { yield &kSky.cloudBurstLights } }
}
