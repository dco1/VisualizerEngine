import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `GIKnobs` — storage lives in IlluminatoramaGIKnobs.swift.
extension IlluminatoramaRenderer {
    public var aovCaptureEnabled: Bool { get { kGI.aovCaptureEnabled } set { kGI.aovCaptureEnabled = newValue } _modify { yield &kGI.aovCaptureEnabled } }
    public internal(set) var aovAccumulatedFrames: Int { get { kGI.aovAccumulatedFrames } set { kGI.aovAccumulatedFrames = newValue } _modify { yield &kGI.aovAccumulatedFrames } }
    public var aoMultiBounce: Bool { get { kGI.aoMultiBounce } set { kGI.aoMultiBounce = newValue } _modify { yield &kGI.aoMultiBounce } }
    public var aoAccumulationScramble: UInt32 { get { kGI.aoAccumulationScramble } set { kGI.aoAccumulationScramble = newValue } _modify { yield &kGI.aoAccumulationScramble } }
    public var aoStillPerFrame: Bool { get { kGI.aoStillPerFrame } set { kGI.aoStillPerFrame = newValue } _modify { yield &kGI.aoStillPerFrame } }
    public var aoAccumulatorOwnsAO: Bool { get { kGI.aoAccumulatorOwnsAO } set { kGI.aoAccumulatorOwnsAO = newValue } _modify { yield &kGI.aoAccumulatorOwnsAO } }
    public var surfCacheIndirectRays: Int { get { kGI.surfCacheIndirectRays } set { kGI.surfCacheIndirectRays = newValue } _modify { yield &kGI.surfCacheIndirectRays } }
    public var surfCacheAlpha: Float { get { kGI.surfCacheAlpha } set { kGI.surfCacheAlpha = newValue } _modify { yield &kGI.surfCacheAlpha } }
    public var surfaceCacheWarmIterations: Int { get { kGI.surfaceCacheWarmIterations } set { kGI.surfaceCacheWarmIterations = newValue } _modify { yield &kGI.surfaceCacheWarmIterations } }
    public var surfaceCacheBackFaceGuard: Bool { get { kGI.surfaceCacheBackFaceGuard } set { kGI.surfaceCacheBackFaceGuard = newValue } _modify { yield &kGI.surfaceCacheBackFaceGuard } }
    public var lightingDefocusShadingMinCoC: Float { get { kGI.lightingDefocusShadingMinCoC } set { kGI.lightingDefocusShadingMinCoC = newValue } _modify { yield &kGI.lightingDefocusShadingMinCoC } }
    public var areaShadowJitterEnabled: Bool { get { kGI.areaShadowJitterEnabled } set { kGI.areaShadowJitterEnabled = newValue } _modify { yield &kGI.areaShadowJitterEnabled } }
    public var debugTerm: DebugTerm { get { kGI.debugTerm } set { kGI.debugTerm = newValue } _modify { yield &kGI.debugTerm } }
    public var ssaoIntensity: Float { get { kGI.ssaoIntensity } set { kGI.ssaoIntensity = newValue } _modify { yield &kGI.ssaoIntensity } }
    public var ssaoRadius: Float { get { kGI.ssaoRadius } set { kGI.ssaoRadius = newValue } _modify { yield &kGI.ssaoRadius } }
    public var ssaoFarRadius: Float { get { kGI.ssaoFarRadius } set { kGI.ssaoFarRadius = newValue } _modify { yield &kGI.ssaoFarRadius } }
    public var ssaoFarIntensity: Float { get { kGI.ssaoFarIntensity } set { kGI.ssaoFarIntensity = newValue } _modify { yield &kGI.ssaoFarIntensity } }
    public var ssrIntensity: Float { get { kGI.ssrIntensity } set { kGI.ssrIntensity = newValue } _modify { yield &kGI.ssrIntensity } }
    public var ssrMaxDistance: Float { get { kGI.ssrMaxDistance } set { kGI.ssrMaxDistance = newValue } _modify { yield &kGI.ssrMaxDistance } }
    public var ssrThickness: Float { get { kGI.ssrThickness } set { kGI.ssrThickness = newValue } _modify { yield &kGI.ssrThickness } }
    public var ssrMaxSteps: UInt32 { get { kGI.ssrMaxSteps } set { kGI.ssrMaxSteps = newValue } _modify { yield &kGI.ssrMaxSteps } }
    public var extendedGBufferEnabled: Bool { get { kGI.extendedGBufferEnabled } set { kGI.extendedGBufferEnabled = newValue } _modify { yield &kGI.extendedGBufferEnabled } }
    public internal(set) var extendedGBufferActive: Bool { get { kGI.extendedGBufferActive } set { kGI.extendedGBufferActive = newValue } _modify { yield &kGI.extendedGBufferActive } }
    public var ssaoSpatialEnabled: Bool { get { kGI.ssaoSpatialEnabled } set { kGI.ssaoSpatialEnabled = newValue } _modify { yield &kGI.ssaoSpatialEnabled } }
    public var ssaoTemporalEnabled: Bool { get { kGI.ssaoTemporalEnabled } set { kGI.ssaoTemporalEnabled = newValue } _modify { yield &kGI.ssaoTemporalEnabled } }
    public var ssaoTemporalBlend: Float { get { kGI.ssaoTemporalBlend } set { kGI.ssaoTemporalBlend = newValue } _modify { yield &kGI.ssaoTemporalBlend } }
    public var ssrTemporalEnabled: Bool { get { kGI.ssrTemporalEnabled } set { kGI.ssrTemporalEnabled = newValue } _modify { yield &kGI.ssrTemporalEnabled } }
    public var ssrTemporalBlend: Float { get { kGI.ssrTemporalBlend } set { kGI.ssrTemporalBlend = newValue } _modify { yield &kGI.ssrTemporalBlend } }
    public var visibilityCullingEnabled: Bool { get { kGI.visibilityCullingEnabled } set { kGI.visibilityCullingEnabled = newValue } _modify { yield &kGI.visibilityCullingEnabled } }
    public var cameraHiddenKinds: Set<MeshKind> { get { kGI.cameraHiddenKinds } set { kGI.cameraHiddenKinds = newValue } _modify { yield &kGI.cameraHiddenKinds } }
    public internal(set) var visibilityCullStats: IlluminatoramaVisibilityCullStats { get { kGI.visibilityCullStats } set { kGI.visibilityCullStats = newValue } _modify { yield &kGI.visibilityCullStats } }
    public var surfaceCacheIncremental: Bool { get { kGI.surfaceCacheIncremental } set { kGI.surfaceCacheIncremental = newValue } _modify { yield &kGI.surfaceCacheIncremental } }
    public var surfaceCacheDenoise: Bool { get { kGI.surfaceCacheDenoise } set { kGI.surfaceCacheDenoise = newValue } _modify { yield &kGI.surfaceCacheDenoise } }
    public var surfaceCacheFeedback: Bool { get { kGI.surfaceCacheFeedback } set { kGI.surfaceCacheFeedback = newValue } _modify { yield &kGI.surfaceCacheFeedback } }
    public var surfCardBudget: Int { get { kGI.surfCardBudget } set { kGI.surfCardBudget = newValue } _modify { yield &kGI.surfCardBudget } }
    public var surfaceCacheStreaming: Bool { get { kGI.surfaceCacheStreaming } set { kGI.surfaceCacheStreaming = newValue } _modify { yield &kGI.surfaceCacheStreaming } }
    public var surfaceCacheLOD: Bool { get { kGI.surfaceCacheLOD } set { kGI.surfaceCacheLOD = newValue } _modify { yield &kGI.surfaceCacheLOD } }
    public internal(set) var surfaceCacheCardlessInstances: Int { get { kGI.surfaceCacheCardlessInstances } set { kGI.surfaceCacheCardlessInstances = newValue } _modify { yield &kGI.surfaceCacheCardlessInstances } }
    public var surfaceCacheMaxTriangles: Int { get { kGI.surfaceCacheMaxTriangles } set { kGI.surfaceCacheMaxTriangles = newValue } _modify { yield &kGI.surfaceCacheMaxTriangles } }
    public var surfaceCachePerTriangleTileSize: Int { get { kGI.surfaceCachePerTriangleTileSize } set { kGI.surfaceCachePerTriangleTileSize = newValue } _modify { yield &kGI.surfaceCachePerTriangleTileSize } }
    public var surfaceCacheCardlessReshade: Bool { get { kGI.surfaceCacheCardlessReshade } set { kGI.surfaceCacheCardlessReshade = newValue } _modify { yield &kGI.surfaceCacheCardlessReshade } }
}
