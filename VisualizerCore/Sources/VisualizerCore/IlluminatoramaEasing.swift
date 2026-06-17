import Foundation

/// How a slider change in the Illuminatorama post-FX panel is applied: instantly,
/// or eased toward the new value over a time constant. `tau` is the exponential-
/// smoothing time constant in seconds for the per-frame update
/// `current += (target - current) · (1 − e^(−dt/tau))` — so `instant` (tau = 0)
/// snaps. Mirrors the camera-easing enum used by the AspectTest scene.
public enum IlluminatoramaEasing: String, CaseIterable, Identifiable, Sendable {
    case instant = "Instant"
    case gentle  = "Gentle"
    case smooth  = "Smooth"
    case slow    = "Slow"
    public var id: String { rawValue }
    public var tau: Double {
        switch self {
        case .instant: return 0
        case .gentle:  return 0.18
        case .smooth:  return 0.40
        case .slow:    return 0.95
        }
    }
}
