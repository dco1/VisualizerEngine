import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `LensKnobs` — storage lives in IlluminatoramaLensKnobs.swift.
extension IlluminatoramaRenderer {
    public var dofEnabled: Bool { get { kLens.dofEnabled } set { kLens.dofEnabled = newValue } _modify { yield &kLens.dofEnabled } }
    public var dofFocusDistance: Float { get { kLens.dofFocusDistance } set { kLens.dofFocusDistance = newValue } _modify { yield &kLens.dofFocusDistance } }
    public var dofCoCCoefficient: Float { get { kLens.dofCoCCoefficient } set { kLens.dofCoCCoefficient = newValue } _modify { yield &kLens.dofCoCCoefficient } }
    public var dofMaxRadius: Float { get { kLens.dofMaxRadius } set { kLens.dofMaxRadius = newValue } _modify { yield &kLens.dofMaxRadius } }
    public var dofBlades: Int { get { kLens.dofBlades } set { kLens.dofBlades = newValue } _modify { yield &kLens.dofBlades } }
    public var dofBladeRotation: Float { get { kLens.dofBladeRotation } set { kLens.dofBladeRotation = newValue } _modify { yield &kLens.dofBladeRotation } }
    public var dofPrefilterScale: Float { get { kLens.dofPrefilterScale } set { kLens.dofPrefilterScale = newValue } _modify { yield &kLens.dofPrefilterScale } }
    public var dofDiffractionCoCPixels: Float { get { kLens.dofDiffractionCoCPixels } set { kLens.dofDiffractionCoCPixels = newValue } _modify { yield &kLens.dofDiffractionCoCPixels } }
    public var dofCatsEye: Float { get { kLens.dofCatsEye } set { kLens.dofCatsEye = newValue } _modify { yield &kLens.dofCatsEye } }
    public var dofSharpSubjects: Bool { get { kLens.dofSharpSubjects } set { kLens.dofSharpSubjects = newValue } _modify { yield &kLens.dofSharpSubjects } }
    public var dofPartialOcclusion: Bool { get { kLens.dofPartialOcclusion } set { kLens.dofPartialOcclusion = newValue } _modify { yield &kLens.dofPartialOcclusion } }
    public var dofFastGather: Bool { get { kLens.dofFastGather } set { kLens.dofFastGather = newValue } _modify { yield &kLens.dofFastGather } }
    public var dofHalfResMinCoC: Float { get { kLens.dofHalfResMinCoC } set { kLens.dofHalfResMinCoC = newValue } _modify { yield &kLens.dofHalfResMinCoC } }
    public var dofQuarterResMinCoC: Float { get { kLens.dofQuarterResMinCoC } set { kLens.dofQuarterResMinCoC = newValue } _modify { yield &kLens.dofQuarterResMinCoC } }
    public var dofReferenceAccumulation: Bool { get { kLens.dofReferenceAccumulation } set { kLens.dofReferenceAccumulation = newValue } _modify { yield &kLens.dofReferenceAccumulation } }
    public var dofReferenceApertureSample: SIMD2<Float> { get { kLens.dofReferenceApertureSample } set { kLens.dofReferenceApertureSample = newValue } _modify { yield &kLens.dofReferenceApertureSample } }
    public var dofReferenceAddsFrame: Bool { get { kLens.dofReferenceAddsFrame } set { kLens.dofReferenceAddsFrame = newValue } _modify { yield &kLens.dofReferenceAddsFrame } }
    public var dofReferenceFrameCount: Int { get { kLens.dofReferenceFrameCount } set { kLens.dofReferenceFrameCount = newValue } _modify { yield &kLens.dofReferenceFrameCount } }
}
