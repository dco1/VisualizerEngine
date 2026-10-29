import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `PostKnobs` — storage lives in IlluminatoramaPostKnobs.swift.
extension IlluminatoramaRenderer {
    public var exposure: Float { get { kPost.exposure } set { kPost.exposure = newValue } _modify { yield &kPost.exposure } }
    public var postFXEasingTau: Double { get { kPost.postFXEasingTau } set { kPost.postFXEasingTau = newValue } _modify { yield &kPost.postFXEasingTau } }
    public var fixedFrameDelta: Double? { get { kPost.fixedFrameDelta } set { kPost.fixedFrameDelta = newValue } _modify { yield &kPost.fixedFrameDelta } }
    public var time: Float { get { kPost.time } set { kPost.time = newValue } _modify { yield &kPost.time } }
    public var selectionOutlineEnabled: Bool { get { kPost.selectionOutlineEnabled } set { kPost.selectionOutlineEnabled = newValue } _modify { yield &kPost.selectionOutlineEnabled } }
    public var selectionOutline: HighlightOutlineStyle { get { kPost.selectionOutline } set { kPost.selectionOutline = newValue } _modify { yield &kPost.selectionOutline } }
    public var hoverOutline: HighlightOutlineStyle { get { kPost.hoverOutline } set { kPost.hoverOutline = newValue } _modify { yield &kPost.hoverOutline } }
    public var highlightMaskInstances: [IlluminatoramaInstance] { get { kPost.highlightMaskInstances } set { kPost.highlightMaskInstances = newValue } _modify { yield &kPost.highlightMaskInstances } }
    public var taaStillAccumulation: Bool { get { kPost.taaStillAccumulation } set { kPost.taaStillAccumulation = newValue } _modify { yield &kPost.taaStillAccumulation } }
    public var taaStillKarisWeights: Bool { get { kPost.taaStillKarisWeights } set { kPost.taaStillKarisWeights = newValue } _modify { yield &kPost.taaStillKarisWeights } }
    public var taaStillSharpen: Bool { get { kPost.taaStillSharpen } set { kPost.taaStillSharpen = newValue } _modify { yield &kPost.taaStillSharpen } }
    public var taaStillCrossfade: Float { get { kPost.taaStillCrossfade } set { kPost.taaStillCrossfade = newValue } _modify { yield &kPost.taaStillCrossfade } }
    /// Freeze the picture currently on screen as `taaStillCrossfade`'s source (taken at the next TAA resolve).
    public func requestTAAStillCrossfadeCapture() { kPost.taaStillCrossfadeCaptureRequested = true }
    public var displayTransform: DisplayTransform { get { kPost.displayTransform } set { kPost.displayTransform = newValue } _modify { yield &kPost.displayTransform } }
    public var projectionShiftNDC: SIMD2<Float> { get { kPost.projectionShiftNDC } set { kPost.projectionShiftNDC = newValue } _modify { yield &kPost.projectionShiftNDC } }
    public var autoExposureHold: Bool { get { kPost.autoExposureHold } set { kPost.autoExposureHold = newValue } _modify { yield &kPost.autoExposureHold } }
    public var exposureMeterGridSize: SIMD2<Int>? { get { kPost.exposureMeterGridSize } set { kPost.exposureMeterGridSize = newValue } _modify { yield &kPost.exposureMeterGridSize } }
    public var autoExposureEnabled: Bool { get { kPost.autoExposureEnabled } set { kPost.autoExposureEnabled = newValue } _modify { yield &kPost.autoExposureEnabled } }
    public var autoExposureTargetEV: Float { get { kPost.autoExposureTargetEV } set { kPost.autoExposureTargetEV = newValue } _modify { yield &kPost.autoExposureTargetEV } }
    public var autoExposureMaxBoost: Float { get { kPost.autoExposureMaxBoost } set { kPost.autoExposureMaxBoost = newValue } _modify { yield &kPost.autoExposureMaxBoost } }
    public var autoExposureMinExposure: Float { get { kPost.autoExposureMinExposure } set { kPost.autoExposureMinExposure = newValue } _modify { yield &kPost.autoExposureMinExposure } }
    public var autoExposureHalfLife: Float { get { kPost.autoExposureHalfLife } set { kPost.autoExposureHalfLife = newValue } _modify { yield &kPost.autoExposureHalfLife } }
    public var autoExposureHighlightProtection: Float { get { kPost.autoExposureHighlightProtection } set { kPost.autoExposureHighlightProtection = newValue } _modify { yield &kPost.autoExposureHighlightProtection } }
    public var autoExposureHighlightEV: Float { get { kPost.autoExposureHighlightEV } set { kPost.autoExposureHighlightEV = newValue } _modify { yield &kPost.autoExposureHighlightEV } }
    public var autoExposureMetering: AutoExposureMetering { get { kPost.autoExposureMetering } set { kPost.autoExposureMetering = newValue } _modify { yield &kPost.autoExposureMetering } }
    public var autoExposureKeyPercentile: Float { get { kPost.autoExposureKeyPercentile } set { kPost.autoExposureKeyPercentile = newValue } _modify { yield &kPost.autoExposureKeyPercentile } }
    public var autoExposureGuardPercentile: Float { get { kPost.autoExposureGuardPercentile } set { kPost.autoExposureGuardPercentile = newValue } _modify { yield &kPost.autoExposureGuardPercentile } }
    public var autoExposureGuardEV: Float { get { kPost.autoExposureGuardEV } set { kPost.autoExposureGuardEV = newValue } _modify { yield &kPost.autoExposureGuardEV } }
    public var exposureHistogramEnabled: Bool { get { kPost.exposureHistogramEnabled } set { kPost.exposureHistogramEnabled = newValue } _modify { yield &kPost.exposureHistogramEnabled } }
    public var scotopicKnee: Float { get { kPost.scotopicKnee } set { kPost.scotopicKnee = newValue } _modify { yield &kPost.scotopicKnee } }
    public var scotopicTint: SIMD3<Float> { get { kPost.scotopicTint } set { kPost.scotopicTint = newValue } _modify { yield &kPost.scotopicTint } }
    public var tonemapSaturation: Float { get { kPost.tonemapSaturation } set { kPost.tonemapSaturation = newValue } _modify { yield &kPost.tonemapSaturation } }
    public var localToneMapping: IlluminatoramaLocalToneMapping { get { kPost.localToneMapping } set { kPost.localToneMapping = newValue } _modify { yield &kPost.localToneMapping } }
    public var whiteBalanceK: Float { get { kPost.whiteBalanceK } set { kPost.whiteBalanceK = newValue } _modify { yield &kPost.whiteBalanceK } }
    public var tint: Float { get { kPost.tint } set { kPost.tint = newValue } _modify { yield &kPost.tint } }
    public var shadows: Float { get { kPost.shadows } set { kPost.shadows = newValue } _modify { yield &kPost.shadows } }
    public var highlights: Float { get { kPost.highlights } set { kPost.highlights = newValue } _modify { yield &kPost.highlights } }
    public var contrast: Float { get { kPost.contrast } set { kPost.contrast = newValue } _modify { yield &kPost.contrast } }
    public var highlightChromaRolloff: Float { get { kPost.highlightChromaRolloff } set { kPost.highlightChromaRolloff = newValue } _modify { yield &kPost.highlightChromaRolloff } }
    public var shadowTemperatureK: Float { get { kPost.shadowTemperatureK } set { kPost.shadowTemperatureK = newValue } _modify { yield &kPost.shadowTemperatureK } }
    public var highlightTemperatureK: Float { get { kPost.highlightTemperatureK } set { kPost.highlightTemperatureK = newValue } _modify { yield &kPost.highlightTemperatureK } }
    public var liveLookMatchStrength: Float { get { kPost.liveLookMatchStrength } set { kPost.liveLookMatchStrength = newValue } _modify { yield &kPost.liveLookMatchStrength } }
    public var liveLookGIDarken: Float { get { kPost.liveLookGIDarken } set { kPost.liveLookGIDarken = newValue } _modify { yield &kPost.liveLookGIDarken } }
    public var liveLookGIWarmth: Float { get { kPost.liveLookGIWarmth } set { kPost.liveLookGIWarmth = newValue } _modify { yield &kPost.liveLookGIWarmth } }
    public var liveLookAODarken: Float { get { kPost.liveLookAODarken } set { kPost.liveLookAODarken = newValue } _modify { yield &kPost.liveLookAODarken } }
    public var antiTilingStrength: Float { get { kPost.antiTilingStrength } set { kPost.antiTilingStrength = newValue } _modify { yield &kPost.antiTilingStrength } }
    public var iblDiffuseSaturation: Float { get { kPost.iblDiffuseSaturation } set { kPost.iblDiffuseSaturation = newValue } _modify { yield &kPost.iblDiffuseSaturation } }
    public var iblDiffuseDesaturation: Float { get { kPost.iblDiffuseDesaturation } set { kPost.iblDiffuseDesaturation = newValue } _modify { yield &kPost.iblDiffuseDesaturation } }
    public var scotopicDesaturation: Float { get { kPost.scotopicDesaturation } set { kPost.scotopicDesaturation = newValue } _modify { yield &kPost.scotopicDesaturation } }
    public var appliesSharedDenoiser: Bool { get { kPost.appliesSharedDenoiser } set { kPost.appliesSharedDenoiser = newValue } _modify { yield &kPost.appliesSharedDenoiser } }
    public var debandDitherEnabled: Bool { get { kPost.debandDitherEnabled } set { kPost.debandDitherEnabled = newValue } _modify { yield &kPost.debandDitherEnabled } }
    public var debandDitherExactSRGB: Bool { get { kPost.debandDitherExactSRGB } set { kPost.debandDitherExactSRGB = newValue } _modify { yield &kPost.debandDitherExactSRGB } }
    public var hdrPreExposure: Float { get { kPost.hdrPreExposure } set { kPost.hdrPreExposure = newValue } _modify { yield &kPost.hdrPreExposure } }
    public var hueStableToe: SIMD2<Float> { get { kPost.hueStableToe } set { kPost.hueStableToe = newValue } _modify { yield &kPost.hueStableToe } }
    public var hueStableToeBareLevel: Bool { get { kPost.hueStableToeBareLevel } set { kPost.hueStableToeBareLevel = newValue } _modify { yield &kPost.hueStableToeBareLevel } }
    public var taaEnabled: Bool { get { kPost.taaEnabled } set { kPost.taaEnabled = newValue } _modify { yield &kPost.taaEnabled } }
    public var sharedTAAOverride: Bool? { get { kPost.sharedTAAOverride } set { kPost.sharedTAAOverride = newValue } _modify { yield &kPost.sharedTAAOverride } }
    public var appliesSharedLensFX: Bool { get { kPost.appliesSharedLensFX } set { kPost.appliesSharedLensFX = newValue } _modify { yield &kPost.appliesSharedLensFX } }
    public var taaHistoryBlend: Float { get { kPost.taaHistoryBlend } set { kPost.taaHistoryBlend = newValue } _modify { yield &kPost.taaHistoryBlend } }
    public var taaJitterPixels: Float { get { kPost.taaJitterPixels } set { kPost.taaJitterPixels = newValue } _modify { yield &kPost.taaJitterPixels } }
    public var reconstructionFilter: IlluminatoramaReconstructionFilter { get { kPost.reconstructionFilter } set { kPost.reconstructionFilter = newValue } _modify { yield &kPost.reconstructionFilter } }
    public var blockingLightingCompile: Bool { get { kPost.blockingLightingCompile } set { kPost.blockingLightingCompile = newValue } _modify { yield &kPost.blockingLightingCompile } }
    public var onFramePresented: ((_ texture: MTLTexture, _ queue: MTLCommandQueue) -> Void)? { get { kPost.onFramePresented } set { kPost.onFramePresented = newValue } _modify { yield &kPost.onFramePresented } }
    public var colorLUTAmount: Float { get { kPost.colorLUTAmount } set { kPost.colorLUTAmount = newValue } _modify { yield &kPost.colorLUTAmount } }
}
