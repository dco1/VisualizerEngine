import Foundation

/// **The per-term output set (AOVs) — the Swift side of `Shaders/IlluminatoramaAOV.h`** (Daydream
/// DH-1304). One named set of lighting terms every lane emits under the same names, units and
/// exposure; the header holds the slice indices the kernels write and the full attribution
/// contract, this enum mirrors them for hosts and tests. `IlluminatoramaAOVTests` fails if the
/// two drift. Shaped so DH-1017 (professional AOV export) can export it unchanged.
///
/// Units: scene-referred linear radiance in the renderer's HDR working units — before exposure,
/// white balance, bloom, display transform and every grade. Sum: Σ `lightingTerms` == `beauty`
/// per pixel, per frame and over the running mean (`IlluminatoramaRenderer.aovCaptureEnabled`).
public enum IlluminatoramaAOV: String, CaseIterable, Sendable {
    // Lighting slices (in the capture texture array, in header order).
    case direct, indirect, emission, environment, specular, transmission, handoff, beauty
    // Geometry / material channels — the G-buffer, not a slice.
    case albedo, normal, depth, roughness, metalness

    /// Where a term lives in the captured frame.
    public enum Storage: Equatable, Sendable {
        /// A slice of `IlluminatoramaRenderer.aovLightingTexture` (rgba32Float, internal size).
        case slice(Int)
        /// A G-buffer target the lighting already reads; no accumulation (a still's G-buffer is
        /// the same every frame).
        case gbuffer
    }

    public var storage: Storage {
        switch self {
        case .direct:       return .slice(0)
        case .indirect:     return .slice(1)
        case .emission:     return .slice(2)
        case .environment:  return .slice(3)
        case .specular:     return .slice(4)
        case .transmission: return .slice(5)
        case .handoff:      return .slice(6)
        case .beauty:       return .slice(7)
        case .albedo, .normal, .depth, .roughness, .metalness: return .gbuffer
        }
    }

    /// The slice index, for a lighting term; nil for a G-buffer channel.
    public var slice: Int? { if case .slice(let i) = storage { return i } else { return nil } }

    /// The terms whose per-pixel sum is `beauty` — `kAOVLightingCount` of them, in slice order.
    public static let lightingTerms: [IlluminatoramaAOV] =
        [.direct, .indirect, .emission, .environment, .specular, .transmission, .handoff]
    /// The number of slices in the capture texture (`kAOVCount`).
    public static let sliceCount = 8

    /// One line of plain language per term — the attribution in the header, for a panel.
    public var summary: String {
        switch self {
        case .direct:       return "Light from the scene's lights (sun, lamps) reaching the surface with no bounce — diffuse and highlight."
        case .indirect:     return "Light that bounced off geometry at least once before the surface (diffuse)."
        case .emission:     return "The surface's own glow."
        case .environment:  return "Sky light reaching the surface with no bounce (diffuse), and the sky itself behind everything."
        case .specular:     return "The surface's reflection of its environment and of the room (not direct-light highlights)."
        case .transmission: return "Light refracted through glass — reserved, 0 until the glass lane is tagged."
        case .handoff:      return "Bookkeeping: a stand-in the deferred pass offered a traced pass to replace, minus what it took — ≈ 0 when the lanes agree."
        case .beauty:       return "The lit picture: the sum of the terms above, before any filter, exposure or tonemap."
        case .albedo:       return "Surface colour (G-buffer)."
        case .normal:       return "Surface normal (G-buffer)."
        case .depth:        return "Distance from the camera (G-buffer)."
        case .roughness:    return "Surface roughness (G-buffer)."
        case .metalness:    return "Metal / dielectric (G-buffer)."
        }
    }

    /// The units every lighting slice is in — the one convention both lanes share.
    public static let units = "scene-referred linear radiance, renderer HDR working units, pre-exposure / pre-tonemap"
}
