import Foundation
import simd

/// How a camera-slider change is applied: instantly, or eased toward the new
/// value over a time constant. `tau` is the exponential-smoothing time constant
/// in seconds (0 = snap).
///
/// Shared by every scene that uses the orbit-camera controls component
/// (`OrbitCameraRig` + `OrbitCameraSection`). Hoisted out of AspectTest so the
/// camera UX is identical across scenes.
public enum CameraEasing: String, CaseIterable, Identifiable, Sendable {
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

/// Pure orbit-camera placement: where the camera sits given an orbit radius,
/// world height, yaw/pitch (in degrees) and the pivot it looks at.
///
/// `yawDeg = 0` puts the camera on +Z looking toward the pivot; increasing yaw
/// orbits clockwise viewed from above. Pitch lifts the camera up and back so the
/// look direction stays pointed at the pivot. `minY` floors the world Y (some
/// scenes keep the camera from dropping through the floor) — pass
/// `-.greatestFiniteMagnitude` for no clamp.
public func orbitCameraPosition(distance: Double, height: Double,
                                 yawDeg: Double, pitchDeg: Double,
                                 pivot: SIMD3<Float>,
                                 minY: Float = -.greatestFiniteMagnitude) -> SIMD3<Float> {
    let dist = Float(distance)
    let yaw = Float(yawDeg) * .pi / 180
    let pitch = Float(pitchDeg) * .pi / 180
    let radial = dist * cos(pitch)
    let x = pivot.x + radial * sin(yaw)
    let z = pivot.z + radial * cos(yaw)
    let y = pivot.y + dist * sin(pitch) + Float(height)
    return SIMD3(x, max(minY, y), z)
}

/// Holds the eased camera scalars for an orbit-style scene and advances them
/// toward the slider targets each tick, so dragging a camera slider glides
/// instead of snapping. The same exponential smoothing AspectTest hand-rolled,
/// extracted so every orbit scene shares one correct implementation.
///
/// Usage each tick:
/// ```
/// rig.advance(distance: s.cameraDistance, height: s.cameraHeight,
///             yawDeg: s.cameraYawDeg, pitchDeg: s.cameraPitchDeg,
///             fovDeg: s.cameraFovDeg, easing: s.cameraEasing, dt: dt)
/// renderer.applyOrbitCamera(rig, pivot: pivot, minY: 0.2)   // (VisualizerRendering)
/// ```
@MainActor
public final class OrbitCameraRig {
    private var _easedDistance: Double?
    private var _easedHeight: Double = 0
    private var _easedYawDeg: Double = 0
    private var _easedPitchDeg: Double = 0
    private var _easedFovDeg: Double = 0

    public init() {}

    /// Advance the eased scalars toward the targets. The FIRST call seeds the
    /// eased state at the current targets (no startup glide); subsequent calls
    /// smooth with `k = 1 − e^(−dt/tau)` (`tau = 0` → snap).
    ///
    /// Because the first call seeds eased == target and static targets never
    /// drift, a scene with unchanging camera sliders renders identically to a
    /// direct `settings.cameraX` write — easing only diverges while a slider is
    /// actively being dragged. (This is why adopting the rig is render-neutral.)
    public func advance(distance: Double, height: Double, yawDeg: Double,
                        pitchDeg: Double, fovDeg: Double,
                        easing: CameraEasing, dt: Double) {
        if _easedDistance == nil {
            _easedDistance = distance
            _easedHeight = height
            _easedYawDeg = yawDeg
            _easedPitchDeg = pitchDeg
            _easedFovDeg = fovDeg
            return
        }
        let tau = easing.tau
        let k = tau <= 0 ? 1.0 : 1 - exp(-dt / tau)
        _easedDistance! += (distance - _easedDistance!) * k
        _easedHeight    += (height   - _easedHeight)    * k
        _easedYawDeg    += (yawDeg    - _easedYawDeg)   * k
        _easedPitchDeg  += (pitchDeg  - _easedPitchDeg) * k
        _easedFovDeg    += (fovDeg    - _easedFovDeg)   * k
    }

    // Eased scalar targets (valid after the first `advance`; equal the seed
    // values on the first frame). Scenes whose camera math is NOT a plain orbit
    // (custom offsets, a moving look-target, env-var angle overrides) read these
    // instead of the raw settings, so the camera still eases while their own
    // position/target formula and framing stay exactly as before.
    public var easedDistance: Double { _easedDistance ?? 0 }
    public var easedHeight: Double { _easedHeight }
    public var easedYawDeg: Double { _easedYawDeg }
    public var easedPitchDeg: Double { _easedPitchDeg }
    public var easedFovDeg: Double { _easedFovDeg }

    /// The eased vertical field of view in radians (for `IlluminatoramaCamera`).
    public var easedFovYRadians: Float { Float(_easedFovDeg) * .pi / 180 }

    /// The eased camera world position for a given pivot, using the STANDARD
    /// orbit math. Valid after the first `advance(...)`. Scenes with custom math
    /// use the eased scalars above instead.
    public func easedPosition(pivot: SIMD3<Float>,
                              minY: Float = -.greatestFiniteMagnitude) -> SIMD3<Float> {
        orbitCameraPosition(distance: _easedDistance ?? 0, height: _easedHeight,
                            yawDeg: _easedYawDeg, pitchDeg: _easedPitchDeg,
                            pivot: pivot, minY: minY)
    }
}
