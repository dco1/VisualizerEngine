import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `FilmKnobs` — storage lives in IlluminatoramaFilmKnobs.swift.
extension IlluminatoramaRenderer {
    public var bloomThreshold: Float { get { kFilm.bloomThreshold } set { kFilm.bloomThreshold = newValue } _modify { yield &kFilm.bloomThreshold } }
    public var bloomIntensity: Float { get { kFilm.bloomIntensity } set { kFilm.bloomIntensity = newValue } _modify { yield &kFilm.bloomIntensity } }
    public var bloomSoftKnee: Float { get { kFilm.bloomSoftKnee } set { kFilm.bloomSoftKnee = newValue } _modify { yield &kFilm.bloomSoftKnee } }
    public var bloomScatter: Float { get { kFilm.bloomScatter } set { kFilm.bloomScatter = newValue } _modify { yield &kFilm.bloomScatter } }
    public var bloomPSFExponent: Float { get { kFilm.bloomPSFExponent } set { kFilm.bloomPSFExponent = newValue } _modify { yield &kFilm.bloomPSFExponent } }
    public var bloomPhysicalGlare: Bool { get { kFilm.bloomPhysicalGlare } set { kFilm.bloomPhysicalGlare = newValue } _modify { yield &kFilm.bloomPhysicalGlare } }
    public var bloomGlareModel: IlluminatoramaGlareSpread.Model { get { kFilm.bloomGlareModel } set { kFilm.bloomGlareModel = newValue } _modify { yield &kFilm.bloomGlareModel } }
    public internal(set) var lastBloomGlareFit: IlluminatoramaGlareSpread.Fit? { get { kFilm.lastBloomGlareFit } set { kFilm.lastBloomGlareFit = newValue } _modify { yield &kFilm.lastBloomGlareFit } }
    public var sensorNoiseStrength: Float { get { kFilm.sensorNoiseStrength } set { kFilm.sensorNoiseStrength = newValue } _modify { yield &kFilm.sensorNoiseStrength } }
    public var sensorElectronsPerUnit: Float { get { kFilm.sensorElectronsPerUnit } set { kFilm.sensorElectronsPerUnit = newValue } _modify { yield &kFilm.sensorElectronsPerUnit } }
    public var sensorReadNoise: Float { get { kFilm.sensorReadNoise } set { kFilm.sensorReadNoise = newValue } _modify { yield &kFilm.sensorReadNoise } }
    public var sensorChannelCorrelation: Float { get { kFilm.sensorChannelCorrelation } set { kFilm.sensorChannelCorrelation = newValue } _modify { yield &kFilm.sensorChannelCorrelation } }
    public var filmDensityGrain: Float { get { kFilm.filmDensityGrain } set { kFilm.filmDensityGrain = newValue } _modify { yield &kFilm.filmDensityGrain } }
    public var filmDensityGrainCorrelation: Float { get { kFilm.filmDensityGrainCorrelation } set { kFilm.filmDensityGrainCorrelation = newValue } _modify { yield &kFilm.filmDensityGrainCorrelation } }
    public var bloomTentRadius: Float { get { kFilm.bloomTentRadius } set { kFilm.bloomTentRadius = newValue } _modify { yield &kFilm.bloomTentRadius } }
    public var lensFlareIntensity: Float { get { kFilm.lensFlareIntensity } set { kFilm.lensFlareIntensity = newValue } _modify { yield &kFilm.lensFlareIntensity } }
    public internal(set) var lastLensFlareParams: SIMD4<Float> { get { kFilm.lastLensFlareParams } set { kFilm.lastLensFlareParams = newValue } _modify { yield &kFilm.lastLensFlareParams } }
    public var chromaticAberration: Float { get { kFilm.chromaticAberration } set { kFilm.chromaticAberration = newValue } _modify { yield &kFilm.chromaticAberration } }
    public var fringe: Float { get { kFilm.fringe } set { kFilm.fringe = newValue } _modify { yield &kFilm.fringe } }
    public var fringeTint: SIMD3<Float> { get { kFilm.fringeTint } set { kFilm.fringeTint = newValue } _modify { yield &kFilm.fringeTint } }
    public var sphericalAberration: Float { get { kFilm.sphericalAberration } set { kFilm.sphericalAberration = newValue } _modify { yield &kFilm.sphericalAberration } }
    public var vignetteStrength: Float { get { kFilm.vignetteStrength } set { kFilm.vignetteStrength = newValue } _modify { yield &kFilm.vignetteStrength } }
    public var vignetteExtent: Float { get { kFilm.vignetteExtent } set { kFilm.vignetteExtent = newValue } _modify { yield &kFilm.vignetteExtent } }
    public var filmGrainStrength: Float { get { kFilm.filmGrainStrength } set { kFilm.filmGrainStrength = newValue } _modify { yield &kFilm.filmGrainStrength } }
    public var filmGrainSize: Float { get { kFilm.filmGrainSize } set { kFilm.filmGrainSize = newValue } _modify { yield &kFilm.filmGrainSize } }
    public var diagramMix: Float { get { kFilm.diagramMix } set { kFilm.diagramMix = newValue } _modify { yield &kFilm.diagramMix } }
    public var diagramWrap: Float { get { kFilm.diagramWrap } set { kFilm.diagramWrap = newValue } _modify { yield &kFilm.diagramWrap } }
    public var diagramOutlineStrength: Float { get { kFilm.diagramOutlineStrength } set { kFilm.diagramOutlineStrength = newValue } _modify { yield &kFilm.diagramOutlineStrength } }
    public var diagramOutlineColor: SIMD3<Float> { get { kFilm.diagramOutlineColor } set { kFilm.diagramOutlineColor = newValue } _modify { yield &kFilm.diagramOutlineColor } }
    public var diagramEdgeDepthSensitivity: Float { get { kFilm.diagramEdgeDepthSensitivity } set { kFilm.diagramEdgeDepthSensitivity = newValue } _modify { yield &kFilm.diagramEdgeDepthSensitivity } }
    public var diagramEdgeNormalSensitivity: Float { get { kFilm.diagramEdgeNormalSensitivity } set { kFilm.diagramEdgeNormalSensitivity = newValue } _modify { yield &kFilm.diagramEdgeNormalSensitivity } }
    public var diagramEdgeThickness: Float { get { kFilm.diagramEdgeThickness } set { kFilm.diagramEdgeThickness = newValue } _modify { yield &kFilm.diagramEdgeThickness } }
    public var halationIntensity: Float { get { kFilm.halationIntensity } set { kFilm.halationIntensity = newValue } _modify { yield &kFilm.halationIntensity } }
    public var halationThreshold: Float { get { kFilm.halationThreshold } set { kFilm.halationThreshold = newValue } _modify { yield &kFilm.halationThreshold } }
    public var halationRadius: Float { get { kFilm.halationRadius } set { kFilm.halationRadius = newValue } _modify { yield &kFilm.halationRadius } }
    public var halationTint: SIMD3<Float> { get { kFilm.halationTint } set { kFilm.halationTint = newValue } _modify { yield &kFilm.halationTint } }
    public var naturalVignetteK: Float { get { kFilm.naturalVignetteK } set { kFilm.naturalVignetteK = newValue } _modify { yield &kFilm.naturalVignetteK } }
    public internal(set) var filmLUTTexture: MTLTexture? { get { kFilm.filmLUTTexture } set { kFilm.filmLUTTexture = newValue } _modify { yield &kFilm.filmLUTTexture } }
    public internal(set) var filmLUTStrength: Float { get { kFilm.filmLUTStrength } set { kFilm.filmLUTStrength = newValue } _modify { yield &kFilm.filmLUTStrength } }
    public internal(set) var filmLUTSize: Float { get { kFilm.filmLUTSize } set { kFilm.filmLUTSize = newValue } _modify { yield &kFilm.filmLUTSize } }
    public var motionBlurStrength: Float { get { kFilm.motionBlurStrength } set { kFilm.motionBlurStrength = newValue } _modify { yield &kFilm.motionBlurStrength } }
    public var motionBlurMaxPx: Float { get { kFilm.motionBlurMaxPx } set { kFilm.motionBlurMaxPx = newValue } _modify { yield &kFilm.motionBlurMaxPx } }
    public var aerialPerspectiveDensity: Float { get { kFilm.aerialPerspectiveDensity } set { kFilm.aerialPerspectiveDensity = newValue } _modify { yield &kFilm.aerialPerspectiveDensity } }
    public var aerialPerspectiveAirlightScale: Float { get { kFilm.aerialPerspectiveAirlightScale } set { kFilm.aerialPerspectiveAirlightScale = newValue } _modify { yield &kFilm.aerialPerspectiveAirlightScale } }
    public var aerialPerspectiveAirlightFromHorizon: Bool { get { kFilm.aerialPerspectiveAirlightFromHorizon } set { kFilm.aerialPerspectiveAirlightFromHorizon = newValue } _modify { yield &kFilm.aerialPerspectiveAirlightFromHorizon } }
    public var sssStrength: Float { get { kFilm.sssStrength } set { kFilm.sssStrength = newValue } _modify { yield &kFilm.sssStrength } }
    public var sssRadius: Float { get { kFilm.sssRadius } set { kFilm.sssRadius = newValue } _modify { yield &kFilm.sssRadius } }
    public var sssTint: SIMD3<Float> { get { kFilm.sssTint } set { kFilm.sssTint = newValue } _modify { yield &kFilm.sssTint } }
    public internal(set) var lastBloomImpulseGPUMilliseconds: Double { get { kFilm.lastBloomImpulseGPUMilliseconds } set { kFilm.lastBloomImpulseGPUMilliseconds = newValue } _modify { yield &kFilm.lastBloomImpulseGPUMilliseconds } }
}
