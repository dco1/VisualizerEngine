import Foundation

/// Which diagnostic "view" the Illuminatorama pipeline presents (issue #65).
///
/// `.composite` is the production image — every other case isolates one part of
/// the pipeline so a flooded / flat / wrong-looking scene can be decomposed:
///
///  • Cases 1–7 mirror `IlluminatoramaRenderer.DebugTerm` deferred-lighting
///    terms (the lighting kernel overwrites the composite with just that term).
///  • 8–9 isolate the RT surface-cache GI / variance (RT + cache scenes only).
///  • 10–15 are raw G-buffer channels, drawn by a debug branch in the tonemap
///    fragment *before* tonemapping, so they read the true stored values.
///
/// The integer `rawValue` is the wire format shared with the renderer's
/// `IlluminatoramaRenderer.DebugTerm` — VisualizerCore can't import
/// VisualizerRendering, so the two enums mirror **by integer**, exactly like
/// `FrameUniforms`. Keep the case→value mapping in lock-step with `DebugTerm`.
///
/// `.composite` (0) is the default and an **exact no-op** for every scene.
public enum IlluminatoramaDebugView: Int, CaseIterable, Identifiable, Sendable {
    case composite = 0
    case directSun = 1
    case pointLights = 2
    case spotLights = 3
    case diffuseIBL = 4
    case specularIBL = 5
    case emission = 6
    case ambient = 7
    case surfaceCacheGI = 8
    case surfaceCacheVariance = 9
    case albedo = 10
    case normal = 11
    case roughness = 12
    case metalness = 13
    case depth = 14
    case velocity = 15

    public var id: Int { rawValue }

    /// Wire value handed to `IlluminatoramaRenderer.debugTerm` (mirror by int).
    public var debugTermRaw: UInt32 { UInt32(rawValue) }

    public var displayName: String {
        switch self {
        case .composite:            return "Final (composite)"
        case .directSun:            return "Direct sun"
        case .pointLights:          return "Point lights"
        case .spotLights:           return "Spot + area lights"
        case .diffuseIBL:           return "Diffuse IBL"
        case .specularIBL:          return "Specular IBL"
        case .emission:             return "Emission"
        case .ambient:              return "Ambient"
        case .surfaceCacheGI:       return "Surface-cache GI"
        case .surfaceCacheVariance: return "Surface-cache variance"
        case .albedo:               return "Albedo"
        case .normal:               return "Normal"
        case .roughness:            return "Roughness"
        case .metalness:            return "Metalness"
        case .depth:                return "Depth"
        case .velocity:             return "Velocity"
        }
    }

    /// Grouping for the panel picker (section headers in the menu).
    public enum Group: String, CaseIterable, Sendable {
        case composite  = "Composite"
        case lighting   = "Lighting terms"
        case rayTracing = "Ray tracing (RT + surface cache)"
        case gbuffer    = "G-buffer channels"
    }

    public var group: Group {
        switch self {
        case .composite:
            return .composite
        case .directSun, .pointLights, .spotLights,
             .diffuseIBL, .specularIBL, .emission, .ambient:
            return .lighting
        case .surfaceCacheGI, .surfaceCacheVariance:
            return .rayTracing
        case .albedo, .normal, .roughness, .metalness, .depth, .velocity:
            return .gbuffer
        }
    }
}
