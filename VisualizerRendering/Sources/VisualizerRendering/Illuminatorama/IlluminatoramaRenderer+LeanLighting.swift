import Foundation
import Metal

/// The G-buffer material classes a scene draws, for lean lighting (`IlluminatoramaRenderer.leanLightingMaterials`).
/// Bits match `kLFHair…kLFCoat` in IlluminatoramaLighting.metal. A class left out is compiled out of the
/// lighting kernel, so a pixel of that class would shade as plain opaque: declare every class the scene draws.
public struct IlluminatoramaLeanMaterials: OptionSet, Sendable, Hashable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    public static let hair   = Self(rawValue: 1 << 16)
    /// Brushed / grained anisotropy (`normalRoughness.a > 1`).
    public static let aniso  = Self(rawValue: 1 << 17)
    /// The sausage-casing glaze band.
    public static let casing = Self(rawValue: 1 << 18)
    /// Cloth sheen (negative emission alpha).
    public static let sheen  = Self(rawValue: 1 << 19)
    /// Clearcoat (positive emission alpha).
    public static let coat   = Self(rawValue: 1 << 20)
    public static let all: Self = [.hair, .aniso, .casing, .sheen, .coat]
    /// Matte, opaque toy pieces: none of the special classes.
    public static let none: Self = []
}

/// **Lean lighting.** `illumi_lighting` is a desktop uber-kernel: dozens of scene features gated by
/// uniforms read at run time. A tile GPU pays for that code even where every branch is skipped, because
/// the compiler sizes the registers for the widest path and occupancy falls (measured on an iPhone 15 Pro:
/// compiling out the layered-BSDF code that `dfgLUTEnabled` keeps alive, though Munimyni never sets its
/// flags, was 32 ms of a 70 ms lighting pass). With `leanLightingEnabled` the renderer specialises the
/// kernel on this frame's own flags and features (function constants 11 and 12), so only reachable code is
/// compiled. A cleared bit means its run-time test was false for every pixel anyway: same pixels, smaller
/// kernel. A change in the scene's features (a lamp switched on) builds a new variant in the background and
/// runs the uber kernel meanwhile, as every other lighting variant does.
extension IlluminatoramaRenderer {

    public var leanLightingEnabled: Bool {
        get { kLighting.leanLightingEnabled }
        set { kLighting.leanLightingEnabled = newValue }
    }
    public var leanLightingMaterials: IlluminatoramaLeanMaterials {
        get { kLighting.leanLightingMaterials }
        set { kLighting.leanLightingMaterials = newValue }
    }

    /// The (flag mask, feature mask) the running lighting variant was specialised on; nil = the uber kernel
    /// (lean lighting off, or its variant still compiling).
    public var activeLeanMasks: SIMD2<UInt32>? { gbs.lastLightingPipeline == nil ? nil : gbs.lastLightingLean }

    /// The kernel's feature mask for these uniforms: a scene bit is set exactly when the kernel's run-time
    /// test for that feature can pass this frame (the tests in IlluminatoramaLighting.metal, `lf(kLF…)`).
    static func leanFeatureMask(_ u: IlluminatoramaFrameUniforms, materials: IlluminatoramaLeanMaterials) -> UInt32 {
        var m: UInt32 = 0
        if u.pointLightCount > 0 { m |= 1 << 0 }
        if u.spotLightCount > 0 { m |= 1 << 1 }
        if u.areaLightCount > 0 { m |= 1 << 2 }
        if u.directionalLightCount > 0 { m |= 1 << 3 }
        if u.contactShadowStrength > 0 { m |= 1 << 4 }
        if u.interiorMask != 0 { m |= 1 << 5 }
        if u.sssStrength > 0 { m |= 1 << 6 }
        if u.aerialPerspectiveDensity > 0 { m |= 1 << 7 }
        if u.extendedGBuffer != 0 { m |= 1 << 8 }
        if u.filmMedium.z > 0.5 { m |= 1 << 9 }
        if u.leafTransmittanceRatio > 0 || u.leafTransmission > 0 { m |= 1 << 10 }
        if u.plushTransmission > 0 || u.plushSheen > 0 { m |= 1 << 11 }
        if u.shadeTransmission > 0 { m |= 1 << 12 }
        return m | (materials.rawValue & IlluminatoramaLeanMaterials.all.rawValue)
    }
}

// MARK: - Lean tonemap

extension IlluminatoramaRenderer {

    /// The tonemap's feature mask for these uniforms: bit n set exactly when effect n's runtime test in
    /// `illumi_tonemap_fs` can pass (`tmf(n)` in IlluminatoramaTonemap.metal).
    static func leanTonemapMask(_ u: IlluminatoramaFrameUniforms) -> UInt32 {
        let on: [Bool] = [
            u.motionBlurStrength != 0,                                       // 0 motion blur / velocity overlay
            u.debugTerm >= 10,                                               // 1 G-buffer debug views
            u.diagramParams.x > 0,                                           // 2 diagram look
            u.chromaticAberration > 0,                                       // 3
            u.sphericalAberration > 0,                                       // 4
            u.halationParams.x > 0,                                          // 5
            u.lensFlareParams.x > 0 && u.lensFlareParams.w > 0,              // 6
            u.liveLookMatchStrength > 0,                                     // 7
            u.naturalVignetteK > 0,                                          // 8
            u.sensorNoise.x > 0 || u.filmMedium.x > 0,                       // 9
            u.displayParams.z > 0,                                           // 10
            u.filmLUTStrength > 0.001,                                       // 11
            u.scotopicDesaturation > 0,                                      // 12
            u.highlightChromaRolloff > 0,                                    // 13
            u.shadowTemperatureK != 6500 || u.highlightTemperatureK != 6500, // 14
            u.fringe > 0,                                                    // 15
            u.vignetteStrength > 0,                                          // 16
            u.filmGrainStrength > 0,                                         // 17
            u.colorLUTAmount > 0,                                            // 18
            u.debandDitherEnabled != 0,                                      // 19
        ]
        var m: UInt32 = 0
        for (i, b) in on.enumerated() where b { m |= 1 << UInt32(i) }
        return m
    }

    /// The frame's tonemap specialised on its own effects (lean lighting on), else nil (the shipped pipeline).
    /// Built once per mask, on first use (a scene's effects rarely change; the first board frames run under
    /// the start screen's warm-up).
    func leanTonemapPipeline() -> MTLRenderPipelineState? {
        guard leanLightingEnabled, let library = engine.library else { return nil }
        let mask = Self.leanTonemapMask(frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self))
        if let p = gbs.leanTonemapPipelines[mask] { return p }
        let cv = MTLFunctionConstantValues()
        var m = mask
        cv.setConstantValue(&m, type: .uint, index: 46)
        let d = MTLRenderPipelineDescriptor()
        d.label = "Illuminatorama.tonemap.lean"
        d.vertexFunction = library.makeFunction(name: "illumi_tonemap_vs")
        guard let fs = try? library.makeFunction(name: "illumi_tonemap_fs", constantValues: cv) else { return nil }
        d.fragmentFunction = fs
        d.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        guard let p = try? device.makeRenderPipelineState(descriptor: d) else { return nil }
        gbs.leanTonemapPipelines[mask] = p
        return p
    }
}
