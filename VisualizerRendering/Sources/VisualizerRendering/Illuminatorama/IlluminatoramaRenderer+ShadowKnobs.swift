import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `ShadowKnobs` — storage lives in IlluminatoramaShadowKnobs.swift.
extension IlluminatoramaRenderer {
    public var spotShadowGPUGeometryOptOutEnabled: Bool { get { kShadow.spotShadowGPUGeometryOptOutEnabled } set { kShadow.spotShadowGPUGeometryOptOutEnabled = newValue } _modify { yield &kShadow.spotShadowGPUGeometryOptOutEnabled } }
    public var shadowsEnabled: Bool { get { kShadow.shadowsEnabled } set { kShadow.shadowsEnabled = newValue } _modify { yield &kShadow.shadowsEnabled } }
    public var directionalShadowExcludedKinds: Set<MeshKind> { get { kShadow.directionalShadowExcludedKinds } set { kShadow.directionalShadowExcludedKinds = newValue } _modify { yield &kShadow.directionalShadowExcludedKinds } }
    public var shadowOnlyMeshKinds: Set<MeshKind> { get { kShadow.shadowOnlyMeshKinds } set { kShadow.shadowOnlyMeshKinds = newValue } _modify { yield &kShadow.shadowOnlyMeshKinds } }
    public var shadowBias: Float { get { kShadow.shadowBias } set { kShadow.shadowBias = newValue } _modify { yield &kShadow.shadowBias } }
    public var shadowSlopeBias: Float { get { kShadow.shadowSlopeBias } set { kShadow.shadowSlopeBias = newValue } _modify { yield &kShadow.shadowSlopeBias } }
    public var shadowCullFollowsDoubleSidedForTest: Bool { get { kShadow.shadowCullFollowsDoubleSidedForTest } set { kShadow.shadowCullFollowsDoubleSidedForTest = newValue } _modify { yield &kShadow.shadowCullFollowsDoubleSidedForTest } }
    public var shadowStoresFrontFaces: Bool { get { kShadow.shadowStoresFrontFaces } set { kShadow.shadowStoresFrontFaces = newValue } _modify { yield &kShadow.shadowStoresFrontFaces } }
    public var shadowRasterBiasOverride: ShadowRasterBias? { get { kShadow.shadowRasterBiasOverride } set { kShadow.shadowRasterBiasOverride = newValue } _modify { yield &kShadow.shadowRasterBiasOverride } }
    public var substrateMeshKinds: Set<MeshKind> { get { kShadow.substrateMeshKinds } set { kShadow.substrateMeshKinds = newValue } _modify { yield &kShadow.substrateMeshKinds } }
    public var substrateDepthBias: Float { get { kShadow.substrateDepthBias } set { kShadow.substrateDepthBias = newValue } _modify { yield &kShadow.substrateDepthBias } }
    public var shadowPcfRadius: UInt32 { get { kShadow.shadowPcfRadius } set { kShadow.shadowPcfRadius = newValue } _modify { yield &kShadow.shadowPcfRadius } }
    public var shadowMaxDistance: Float { get { kShadow.shadowMaxDistance } set { kShadow.shadowMaxDistance = newValue } _modify { yield &kShadow.shadowMaxDistance } }
    public var spotShadowsEnabled: Bool { get { kShadow.spotShadowsEnabled } set { kShadow.spotShadowsEnabled = newValue } _modify { yield &kShadow.spotShadowsEnabled } }
    public var spotShadowBias: Float { get { kShadow.spotShadowBias } set { kShadow.spotShadowBias = newValue } _modify { yield &kShadow.spotShadowBias } }
    public var pointShadowsEnabled: Bool { get { kShadow.pointShadowsEnabled } set { kShadow.pointShadowsEnabled = newValue } _modify { yield &kShadow.pointShadowsEnabled } }
    public var pointShadowBias: Float { get { kShadow.pointShadowBias } set { kShadow.pointShadowBias = newValue } _modify { yield &kShadow.pointShadowBias } }
    public var contactShadowStrength: Float { get { kShadow.contactShadowStrength } set { kShadow.contactShadowStrength = newValue } _modify { yield &kShadow.contactShadowStrength } }
    public var contactShadowLength: Float { get { kShadow.contactShadowLength } set { kShadow.contactShadowLength = newValue } _modify { yield &kShadow.contactShadowLength } }
    public var contactShadowSteps: UInt32 { get { kShadow.contactShadowSteps } set { kShadow.contactShadowSteps = newValue } _modify { yield &kShadow.contactShadowSteps } }
    public var contactShadowThickness: Float { get { kShadow.contactShadowThickness } set { kShadow.contactShadowThickness = newValue } _modify { yield &kShadow.contactShadowThickness } }
    public internal(set) var spotLightProfileRowCount: Int { get { kShadow.spotLightProfileRowCount } set { kShadow.spotLightProfileRowCount = newValue } _modify { yield &kShadow.spotLightProfileRowCount } }
}
