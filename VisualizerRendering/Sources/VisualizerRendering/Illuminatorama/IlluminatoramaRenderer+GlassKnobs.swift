import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `GlassKnobs` — storage lives in IlluminatoramaGlassKnobs.swift.
extension IlluminatoramaRenderer {
    public var glassPaneKind: MeshKind? { get { kGlass.glassPaneKind } set { kGlass.glassPaneKind = newValue } _modify { yield &kGlass.glassPaneKind } }
    public var glassPaneInstance: IlluminatoramaGlassInstance? { get { kGlass.glassPaneInstance } set { kGlass.glassPaneInstance = newValue } _modify { yield &kGlass.glassPaneInstance } }
    public var glassInstances: [IlluminatoramaGlassInstance] { get { kGlass.glassInstances } set { kGlass.glassInstances = newValue } _modify { yield &kGlass.glassInstances } }
    public var glassMeshGroups: [GlassMeshGroup] { get { kGlass.glassMeshGroups } set { kGlass.glassMeshGroups = newValue } _modify { yield &kGlass.glassMeshGroups } }
    public var glassHitTexturingEnabled: Bool { get { kGlass.glassHitTexturingEnabled } set { kGlass.glassHitTexturingEnabled = newValue } _modify { yield &kGlass.glassHitTexturingEnabled } }
    public var glassHitLocalLightsEnabled: Bool { get { kGlass.glassHitLocalLightsEnabled } set { kGlass.glassHitLocalLightsEnabled = newValue } _modify { yield &kGlass.glassHitLocalLightsEnabled } }
    public var glassHitInteriorFillEnabled: Bool { get { kGlass.glassHitInteriorFillEnabled } set { kGlass.glassHitInteriorFillEnabled = newValue } _modify { yield &kGlass.glassHitInteriorFillEnabled } }
    public var secondaryAreaLightsEnabled: Bool { get { kGlass.secondaryAreaLightsEnabled } set { kGlass.secondaryAreaLightsEnabled = newValue } _modify { yield &kGlass.secondaryAreaLightsEnabled } }
    public var secondaryAreaShadowRays: Int { get { kGlass.secondaryAreaShadowRays } set { kGlass.secondaryAreaShadowRays = newValue } _modify { yield &kGlass.secondaryAreaShadowRays } }
    public var glassCheapMode: Int { get { kGlass.glassCheapMode } set { kGlass.glassCheapMode = newValue } _modify { yield &kGlass.glassCheapMode } }
    public var thinFilmStrength: Float { get { kGlass.thinFilmStrength } set { kGlass.thinFilmStrength = newValue } _modify { yield &kGlass.thinFilmStrength } }
    public var thinFilmThicknessNm: Float { get { kGlass.thinFilmThicknessNm } set { kGlass.thinFilmThicknessNm = newValue } _modify { yield &kGlass.thinFilmThicknessNm } }
    public var thinFilmIOR: Float { get { kGlass.thinFilmIOR } set { kGlass.thinFilmIOR = newValue } _modify { yield &kGlass.thinFilmIOR } }
    public var bubbleWobbleAmp: Float { get { kGlass.bubbleWobbleAmp } set { kGlass.bubbleWobbleAmp = newValue } _modify { yield &kGlass.bubbleWobbleAmp } }
    public var bubbleWobbleFreq: Float { get { kGlass.bubbleWobbleFreq } set { kGlass.bubbleWobbleFreq = newValue } _modify { yield &kGlass.bubbleWobbleFreq } }
    public var causticsEnabled: Bool { get { kGlass.causticsEnabled } set { kGlass.causticsEnabled = newValue } _modify { yield &kGlass.causticsEnabled } }
    public var causticPhotonCount: Int { get { kGlass.causticPhotonCount } set { kGlass.causticPhotonCount = newValue } _modify { yield &kGlass.causticPhotonCount } }
    public var causticStrength: Float { get { kGlass.causticStrength } set { kGlass.causticStrength = newValue } _modify { yield &kGlass.causticStrength } }
    public var causticDecay: Float { get { kGlass.causticDecay } set { kGlass.causticDecay = newValue } _modify { yield &kGlass.causticDecay } }
    public var causticPhotonEnergy: Float { get { kGlass.causticPhotonEnergy } set { kGlass.causticPhotonEnergy = newValue } _modify { yield &kGlass.causticPhotonEnergy } }
}
