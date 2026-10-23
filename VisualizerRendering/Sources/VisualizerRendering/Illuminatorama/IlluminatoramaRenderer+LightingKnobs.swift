import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `LightingKnobs` — storage lives in IlluminatoramaLightingKnobs.swift.
extension IlluminatoramaRenderer {
    public var instances: [InstanceRef] { get { kLighting.instances } set { kLighting.instances = newValue } _modify { yield &kLighting.instances } }
    public internal(set) var staticSkipStats: StaticSkipStats { get { kLighting.staticSkipStats } set { kLighting.staticSkipStats = newValue } _modify { yield &kLighting.staticSkipStats } }
    public var onEncodeFrameCompute: ((_ cb: MTLCommandBuffer) -> Void)? { get { kLighting.onEncodeFrameCompute } set { kLighting.onEncodeFrameCompute = newValue } _modify { yield &kLighting.onEncodeFrameCompute } }
    public var onEncodeAfterGBuffer: ((_ cb: MTLCommandBuffer) -> Void)? { get { kLighting.onEncodeAfterGBuffer } set { kLighting.onEncodeAfterGBuffer = newValue } _modify { yield &kLighting.onEncodeAfterGBuffer } }
    public var pointLights: [IlluminatoramaPointLight] { get { kLighting.pointLights } set { kLighting.pointLights = newValue } _modify { yield &kLighting.pointLights } }
    public var spotLights: [IlluminatoramaSpotLight] { get { kLighting.spotLights } set { kLighting.spotLights = newValue } _modify { yield &kLighting.spotLights } }
    public var areaLights: [IlluminatoramaAreaLight] { get { kLighting.areaLights } set { kLighting.areaLights = newValue } _modify { yield &kLighting.areaLights } }
    public var extraDirectionals: [IlluminatoramaDirectionalLight] { get { kLighting.extraDirectionals } set { kLighting.extraDirectionals = newValue } _modify { yield &kLighting.extraDirectionals } }
    public var directionalLightDirection: SIMD3<Float> { get { kLighting.directionalLightDirection } set { kLighting.directionalLightDirection = newValue } _modify { yield &kLighting.directionalLightDirection } }
    public var directionalLightColor: SIMD3<Float> { get { kLighting.directionalLightColor } set { kLighting.directionalLightColor = newValue } _modify { yield &kLighting.directionalLightColor } }
    public var ambientColor: SIMD3<Float> { get { kLighting.ambientColor } set { kLighting.ambientColor = newValue } _modify { yield &kLighting.ambientColor } }
    public var leafTransmission: Float { get { kLighting.leafTransmission } set { kLighting.leafTransmission = newValue } _modify { yield &kLighting.leafTransmission } }
    public var leafTransmittanceRatio: Float { get { kLighting.leafTransmittanceRatio } set { kLighting.leafTransmittanceRatio = newValue } _modify { yield &kLighting.leafTransmittanceRatio } }
    public var plushSheen: Float { get { kLighting.plushSheen } set { kLighting.plushSheen = newValue } _modify { yield &kLighting.plushSheen } }
    public var plushTransmission: Float { get { kLighting.plushTransmission } set { kLighting.plushTransmission = newValue } _modify { yield &kLighting.plushTransmission } }
    public var shadeTransmission: Float { get { kLighting.shadeTransmission } set { kLighting.shadeTransmission = newValue } _modify { yield &kLighting.shadeTransmission } }
    public var interiorMappedMeshKinds: [MeshKind: IlluminatoramaInteriorMapping] { get { kLighting.interiorMappedMeshKinds } set { kLighting.interiorMappedMeshKinds = newValue } _modify { yield &kLighting.interiorMappedMeshKinds } }
    public var interiorLayerMask: UInt32 { get { kLighting.interiorLayerMask } set { kLighting.interiorLayerMask = newValue } _modify { yield &kLighting.interiorLayerMask } }
    public var interiorIBLUp: Float { get { kLighting.interiorIBLUp } set { kLighting.interiorIBLUp = newValue } _modify { yield &kLighting.interiorIBLUp } }
    public var interiorIBLSide: Float { get { kLighting.interiorIBLSide } set { kLighting.interiorIBLSide = newValue } _modify { yield &kLighting.interiorIBLSide } }
    public var interiorAmbient: Float { get { kLighting.interiorAmbient } set { kLighting.interiorAmbient = newValue } _modify { yield &kLighting.interiorAmbient } }
    public var interiorIrradianceUp: SIMD3<Float> { get { kLighting.interiorIrradianceUp } set { kLighting.interiorIrradianceUp = newValue } _modify { yield &kLighting.interiorIrradianceUp } }
    public var interiorIrradianceSide: SIMD3<Float> { get { kLighting.interiorIrradianceSide } set { kLighting.interiorIrradianceSide = newValue } _modify { yield &kLighting.interiorIrradianceSide } }
    public var interiorIrradianceDown: SIMD3<Float> { get { kLighting.interiorIrradianceDown } set { kLighting.interiorIrradianceDown = newValue } _modify { yield &kLighting.interiorIrradianceDown } }
    public var interiorIrradianceWeight: Float { get { kLighting.interiorIrradianceWeight } set { kLighting.interiorIrradianceWeight = newValue } _modify { yield &kLighting.interiorIrradianceWeight } }
    public var interiorApertures: [IlluminatoramaInteriorAperture] { get { kLighting.interiorApertures } set { kLighting.interiorApertures = newValue } _modify { yield &kLighting.interiorApertures } }
    public var interiorApertureGradation: Float { get { kLighting.interiorApertureGradation } set { kLighting.interiorApertureGradation = newValue } _modify { yield &kLighting.interiorApertureGradation } }
    public var interiorRoomGains: [Float] { get { kLighting.interiorRoomGains } set { kLighting.interiorRoomGains = newValue } _modify { yield &kLighting.interiorRoomGains } }
    public var interiorRoomGainsEnabled: Bool { get { kLighting.interiorRoomGainsEnabled } set { kLighting.interiorRoomGainsEnabled = newValue } _modify { yield &kLighting.interiorRoomGainsEnabled } }
    public var areaLTCOverride: Bool? { get { kLighting.areaLTCOverride } set { kLighting.areaLTCOverride = newValue } _modify { yield &kLighting.areaLTCOverride } }
    public var iblIntensity: Float { get { kLighting.iblIntensity } set { kLighting.iblIntensity = newValue } _modify { yield &kLighting.iblIntensity } }
    public var iblEnabled: Bool { get { kLighting.iblEnabled } set { kLighting.iblEnabled = newValue } _modify { yield &kLighting.iblEnabled } }
    public var equirectSky: MTLTexture? { get { kLighting.equirectSky } set { kLighting.equirectSky = newValue } _modify { yield &kLighting.equirectSky } }
    public var iblNeedsRebake: Bool { get { kLighting.iblNeedsRebake } set { kLighting.iblNeedsRebake = newValue } _modify { yield &kLighting.iblNeedsRebake } }
    public var iblRebakeInterval: Float { get { kLighting.iblRebakeInterval } set { kLighting.iblRebakeInterval = newValue } _modify { yield &kLighting.iblRebakeInterval } }
    public var dfgLUTEnabled: Bool { get { kLighting.dfgLUTEnabled } set { kLighting.dfgLUTEnabled = newValue } _modify { yield &kLighting.dfgLUTEnabled } }
    public var layeredMultiscatter: Bool { get { kLighting.layeredMultiscatter } set { kLighting.layeredMultiscatter = newValue } _modify { yield &kLighting.layeredMultiscatter } }
    public var diffuseRoughnessEnabled: Bool { get { kLighting.diffuseRoughnessEnabled } set { kLighting.diffuseRoughnessEnabled = newValue } _modify { yield &kLighting.diffuseRoughnessEnabled } }
    public var layeredClothSheen: Bool { get { kLighting.layeredClothSheen } set { kLighting.layeredClothSheen = newValue } _modify { yield &kLighting.layeredClothSheen } }
    public internal(set) var initTiming: IlluminatoramaInitTiming { get { kLighting.initTiming } set { kLighting.initTiming = newValue } _modify { yield &kLighting.initTiming } }
}
