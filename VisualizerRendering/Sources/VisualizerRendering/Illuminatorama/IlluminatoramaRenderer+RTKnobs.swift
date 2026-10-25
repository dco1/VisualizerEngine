import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `RTKnobs` — storage lives in IlluminatoramaRTKnobs.swift.
extension IlluminatoramaRenderer {
    public var rtEnabled: Bool { get { kRT.rtEnabled } set { kRT.rtEnabled = newValue } _modify { yield &kRT.rtEnabled } }
    public var rtOpaqueLightingEnabled: Bool { get { kRT.rtOpaqueLightingEnabled } set { kRT.rtOpaqueLightingEnabled = newValue } _modify { yield &kRT.rtOpaqueLightingEnabled } }
    public var rtSunDirection: SIMD3<Float> { get { kRT.rtSunDirection } set { kRT.rtSunDirection = newValue } _modify { yield &kRT.rtSunDirection } }
    public var rtSunColor: SIMD3<Float> { get { kRT.rtSunColor } set { kRT.rtSunColor = newValue } _modify { yield &kRT.rtSunColor } }
    public var rtSunSoftnessRad: Float { get { kRT.rtSunSoftnessRad } set { kRT.rtSunSoftnessRad = newValue } _modify { yield &kRT.rtSunSoftnessRad } }
    public var rtSunSoftShadowsEnabled: Bool { get { kRT.rtSunSoftShadowsEnabled } set { kRT.rtSunSoftShadowsEnabled = newValue } _modify { yield &kRT.rtSunSoftShadowsEnabled } }
    public var rtGIStrength: Float { get { kRT.rtGIStrength } set { kRT.rtGIStrength = newValue } _modify { yield &kRT.rtGIStrength } }
    public var rtGIReplacesDiffuseSky: Bool { get { kRT.rtGIReplacesDiffuseSky } set { kRT.rtGIReplacesDiffuseSky = newValue } _modify { yield &kRT.rtGIReplacesDiffuseSky } }
    public var rtGIOwnsRoomFill: GIRoomFillHandoff { get { kRT.rtGIOwnsRoomFill } set { kRT.rtGIOwnsRoomFill = newValue } _modify { yield &kRT.rtGIOwnsRoomFill } }
    public var rtGIHitFill: Bool { get { kRT.rtGIHitFill } set { kRT.rtGIHitFill = newValue } _modify { yield &kRT.rtGIHitFill } }
    public var rtPathTracedGI: Bool { get { kRT.rtPathTracedGI } set { kRT.rtPathTracedGI = newValue } _modify { yield &kRT.rtPathTracedGI } }
    public var rtPathBounces: Int { get { kRT.rtPathBounces } set { kRT.rtPathBounces = newValue } _modify { yield &kRT.rtPathBounces } }
    public var rtPathClamp: Float { get { kRT.rtPathClamp } set { kRT.rtPathClamp = newValue } _modify { yield &kRT.rtPathClamp } }
    public var rtPathLightSampling: Bool { get { kRT.rtPathLightSampling } set { kRT.rtPathLightSampling = newValue } _modify { yield &kRT.rtPathLightSampling } }
    public var rtPathLampShadows: Bool { get { kRT.rtPathLampShadows } set { kRT.rtPathLampShadows = newValue } _modify { yield &kRT.rtPathLampShadows } }
    public var rtPathDebugFlags: UInt32 { get { kRT.rtPathDebugFlags } set { kRT.rtPathDebugFlags = newValue } _modify { yield &kRT.rtPathDebugFlags } }
    public var rtPathLampMinDistance: Float { get { kRT.rtPathLampMinDistance } set { kRT.rtPathLampMinDistance = newValue } _modify { yield &kRT.rtPathLampMinDistance } }
    public var rtPathLampSourceRadius: Float { get { kRT.rtPathLampSourceRadius } set { kRT.rtPathLampSourceRadius = newValue } _modify { yield &kRT.rtPathLampSourceRadius } }
    public var rtPathLampBulbs: Bool { get { kRT.rtPathLampBulbs } set { kRT.rtPathLampBulbs = newValue } _modify { yield &kRT.rtPathLampBulbs } }
    public var rtShadowRays: Int { get { kRT.rtShadowRays } set { kRT.rtShadowRays = newValue } _modify { yield &kRT.rtShadowRays } }
    public var rtAreaShadowRays: Int { get { kRT.rtAreaShadowRays } set { kRT.rtAreaShadowRays = newValue } _modify { yield &kRT.rtAreaShadowRays } }
    public var rtStillCapsRelaxed: Bool { get { kRT.rtStillCapsRelaxed } set { kRT.rtStillCapsRelaxed = newValue } _modify { yield &kRT.rtStillCapsRelaxed } }
    public var rtaoEnabled: Bool { get { kRT.rtaoEnabled } set { kRT.rtaoEnabled = newValue } _modify { yield &kRT.rtaoEnabled } }
    public var rtaoIntensity: Float { get { kRT.rtaoIntensity } set { kRT.rtaoIntensity = newValue } _modify { yield &kRT.rtaoIntensity } }
    public var rtaoRadius: Float { get { kRT.rtaoRadius } set { kRT.rtaoRadius = newValue } _modify { yield &kRT.rtaoRadius } }
    public var rtaoRays: Int { get { kRT.rtaoRays } set { kRT.rtaoRays = newValue } _modify { yield &kRT.rtaoRays } }
    public var rtaoDidRunLastFrame: Bool { get { kRT.rtaoDidRunLastFrame } set { kRT.rtaoDidRunLastFrame = newValue } _modify { yield &kRT.rtaoDidRunLastFrame } }
    public var rtGIRays: Int { get { kRT.rtGIRays } set { kRT.rtGIRays = newValue } _modify { yield &kRT.rtGIRays } }
    public var rtSpecStrength: Float { get { kRT.rtSpecStrength } set { kRT.rtSpecStrength = newValue } _modify { yield &kRT.rtSpecStrength } }
    public internal(set) var rtSupported: Bool { get { kRT.rtSupported } set { kRT.rtSupported = newValue } _modify { yield &kRT.rtSupported } }
    public var rtSunOwnership: RTSunOwnership { get { kRT.rtSunOwnership } set { kRT.rtSunOwnership = newValue } _modify { yield &kRT.rtSunOwnership } }
    public var rtDenoiseEnabled: Bool { get { kRT.rtDenoiseEnabled } set { kRT.rtDenoiseEnabled = newValue } _modify { yield &kRT.rtDenoiseEnabled } }
    public var rtDenoiseRadius: Int { get { kRT.rtDenoiseRadius } set { kRT.rtDenoiseRadius = newValue } _modify { yield &kRT.rtDenoiseRadius } }
    public var rtDenoiseDepthSensitivity: Float { get { kRT.rtDenoiseDepthSensitivity } set { kRT.rtDenoiseDepthSensitivity = newValue } _modify { yield &kRT.rtDenoiseDepthSensitivity } }
    public var rtDenoiseNormalSharpness: Float { get { kRT.rtDenoiseNormalSharpness } set { kRT.rtDenoiseNormalSharpness = newValue } _modify { yield &kRT.rtDenoiseNormalSharpness } }
    public var rtGITemporalEnabled: Bool { get { kRT.rtGITemporalEnabled } set { kRT.rtGITemporalEnabled = newValue } _modify { yield &kRT.rtGITemporalEnabled } }
    public var rtGITemporalBlend: Float { get { kRT.rtGITemporalBlend } set { kRT.rtGITemporalBlend = newValue } _modify { yield &kRT.rtGITemporalBlend } }
    public var rtGIStaticAccumulation: Bool { get { kRT.rtGIStaticAccumulation } set { kRT.rtGIStaticAccumulation = newValue } _modify { yield &kRT.rtGIStaticAccumulation } }
    public var rtGIStillPerFrame: Bool { get { kRT.rtGIStillPerFrame } set { kRT.rtGIStillPerFrame = newValue } _modify { yield &kRT.rtGIStillPerFrame } }
    public var rtGITemporalClamp: Float { get { kRT.rtGITemporalClamp } set { kRT.rtGITemporalClamp = newValue } _modify { yield &kRT.rtGITemporalClamp } }
    public var rtPathRadianceCacheCell: Float { get { kRT.rtPathRadianceCacheCell } set { kRT.rtPathRadianceCacheCell = newValue } _modify { yield &kRT.rtPathRadianceCacheCell } }
    public var rtPathRadianceCacheBuildFrames: Int { get { kRT.rtPathRadianceCacheBuildFrames } set { kRT.rtPathRadianceCacheBuildFrames = newValue } _modify { yield &kRT.rtPathRadianceCacheBuildFrames } }
    public var rtPathRadianceCacheBuildRays: Int { get { kRT.rtPathRadianceCacheBuildRays } set { kRT.rtPathRadianceCacheBuildRays = newValue } _modify { yield &kRT.rtPathRadianceCacheBuildRays } }
    public var svgfStillLuminanceStop: Bool { get { kRT.svgfStillLuminanceStop } set { kRT.svgfStillLuminanceStop = newValue } _modify { yield &kRT.svgfStillLuminanceStop } }
    public var rtReflectionsEnabled: Bool { get { kRT.rtReflectionsEnabled } set { kRT.rtReflectionsEnabled = newValue } _modify { yield &kRT.rtReflectionsEnabled } }
    public var rtTLASForReflections: Bool { get { kRT.rtTLASForReflections } set { kRT.rtTLASForReflections = newValue } _modify { yield &kRT.rtTLASForReflections } }
    public var rtReflStrength: Float { get { kRT.rtReflStrength } set { kRT.rtReflStrength = newValue } _modify { yield &kRT.rtReflStrength } }
    public var rtReflMaxDistance: Float { get { kRT.rtReflMaxDistance } set { kRT.rtReflMaxDistance = newValue } _modify { yield &kRT.rtReflMaxDistance } }
    public var rtReflRoughnessCutoff: Float { get { kRT.rtReflRoughnessCutoff } set { kRT.rtReflRoughnessCutoff = newValue } _modify { yield &kRT.rtReflRoughnessCutoff } }
    public var rtReflRays: Int { get { kRT.rtReflRays } set { kRT.rtReflRays = newValue } _modify { yield &kRT.rtReflRays } }
    public var rtSurfaceSnapEnabled: Bool { get { kRT.rtSurfaceSnapEnabled } set { kRT.rtSurfaceSnapEnabled = newValue } _modify { yield &kRT.rtSurfaceSnapEnabled } }
    public var rtGlassEnabled: Bool { get { kRT.rtGlassEnabled } set { kRT.rtGlassEnabled = newValue } _modify { yield &kRT.rtGlassEnabled } }
    public var rtGlassMaxBounces: UInt32 { get { kRT.rtGlassMaxBounces } set { kRT.rtGlassMaxBounces = newValue } _modify { yield &kRT.rtGlassMaxBounces } }
    public var rtGlassShadowRays: UInt32 { get { kRT.rtGlassShadowRays } set { kRT.rtGlassShadowRays = newValue } _modify { yield &kRT.rtGlassShadowRays } }
    public var rtGlassDispersionEnabled: Bool { get { kRT.rtGlassDispersionEnabled } set { kRT.rtGlassDispersionEnabled = newValue } _modify { yield &kRT.rtGlassDispersionEnabled } }
    public var rtGlassScreenSpaceTransmission: Bool { get { kRT.rtGlassScreenSpaceTransmission } set { kRT.rtGlassScreenSpaceTransmission = newValue } _modify { yield &kRT.rtGlassScreenSpaceTransmission } }
    public var rtGlassDefocusTraceFactor: Int { get { kRT.rtGlassDefocusTraceFactor } set { kRT.rtGlassDefocusTraceFactor = newValue } _modify { yield &kRT.rtGlassDefocusTraceFactor } }
    public var rtGlassDefocusMinCoCTexels: Float { get { kRT.rtGlassDefocusMinCoCTexels } set { kRT.rtGlassDefocusMinCoCTexels = newValue } _modify { yield &kRT.rtGlassDefocusMinCoCTexels } }
    public var svgfEnabled: Bool { get { kRT.svgfEnabled } set { kRT.svgfEnabled = newValue } _modify { yield &kRT.svgfEnabled } }
    public var svgfStillForced: Bool { get { kRT.svgfStillForced } set { kRT.svgfStillForced = newValue } _modify { yield &kRT.svgfStillForced } }
    public var svgfStillLevels: Int { get { kRT.svgfStillLevels } set { kRT.svgfStillLevels = newValue } _modify { yield &kRT.svgfStillLevels } }
    public var svgfDemodulateAlbedo: Bool { get { kRT.svgfDemodulateAlbedo } set { kRT.svgfDemodulateAlbedo = newValue } _modify { yield &kRT.svgfDemodulateAlbedo } }
    public var svgfLevels: Int { get { kRT.svgfLevels } set { kRT.svgfLevels = newValue } _modify { yield &kRT.svgfLevels } }
    public var svgfSigmaL: Float { get { kRT.svgfSigmaL } set { kRT.svgfSigmaL = newValue } _modify { yield &kRT.svgfSigmaL } }
    public var svgfSigmaZ: Float { get { kRT.svgfSigmaZ } set { kRT.svgfSigmaZ = newValue } _modify { yield &kRT.svgfSigmaZ } }
    public var svgfSigmaN: Float { get { kRT.svgfSigmaN } set { kRT.svgfSigmaN = newValue } _modify { yield &kRT.svgfSigmaN } }
    public var svgfLumFloor: Float { get { kRT.svgfLumFloor } set { kRT.svgfLumFloor = newValue } _modify { yield &kRT.svgfLumFloor } }
    public internal(set) var rtStaleBLASReuseCount: Int { get { kRT.rtStaleBLASReuseCount } set { kRT.rtStaleBLASReuseCount = newValue } _modify { yield &kRT.rtStaleBLASReuseCount } }
}
