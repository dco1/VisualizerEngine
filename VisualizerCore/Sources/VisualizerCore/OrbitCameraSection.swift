import SwiftUI

/// Reusable **Camera** settings section — the six orbit sliders
/// (distance / height / yaw / pitch / FOV) plus the easing picker in the header.
///
/// Every orbit-style scene used to hand-write this block (and a private
/// `LabeledSlider` per scene). This is the one shared version, so the camera UI
/// is identical everywhere and pairs with `OrbitCameraRig` /
/// `IlluminatoramaRenderer.applyOrbitCamera`. Bind it to the scene's own settings
/// fields:
/// ```
/// OrbitCameraSection(distance: $settings.cameraDistance,
///                    height: $settings.cameraHeight,
///                    yawDeg: $settings.cameraYawDeg,
///                    pitchDeg: $settings.cameraPitchDeg,
///                    fovDeg: $settings.cameraFovDeg,
///                    easing: $settings.cameraEasing)
/// ```
/// Ranges default to the AspectTest values; override per scene as needed. Sliders
/// honour ⇧-shift snap (via `shiftSnapped`) like the rest of the app.
public struct OrbitCameraSection: View {
    @Binding var distance: Double
    @Binding var height: Double
    @Binding var yawDeg: Double
    @Binding var pitchDeg: Double
    @Binding var fovDeg: Double
    @Binding var easing: CameraEasing

    let distanceRange: ClosedRange<Double>
    let heightRange: ClosedRange<Double>
    let yawRange: ClosedRange<Double>
    let pitchRange: ClosedRange<Double>
    let fovRange: ClosedRange<Double>

    public init(distance: Binding<Double>,
                height: Binding<Double>,
                yawDeg: Binding<Double>,
                pitchDeg: Binding<Double>,
                fovDeg: Binding<Double>,
                easing: Binding<CameraEasing>,
                distanceRange: ClosedRange<Double> = 1.5...12.0,
                heightRange: ClosedRange<Double> = 0.0...4.0,
                yawRange: ClosedRange<Double> = -180...180,
                pitchRange: ClosedRange<Double> = -10...85,
                fovRange: ClosedRange<Double> = 15...75) {
        self._distance = distance
        self._height = height
        self._yawDeg = yawDeg
        self._pitchDeg = pitchDeg
        self._fovDeg = fovDeg
        self._easing = easing
        self.distanceRange = distanceRange
        self.heightRange = heightRange
        self.yawRange = yawRange
        self.pitchRange = pitchRange
        self.fovRange = fovRange
    }

    public var body: some View {
        Section {
            slider("Distance", $distance, distanceRange, "%.1f m")
            slider("Height", $height, heightRange, "%.2f m")
            slider("Yaw", $yawDeg, yawRange, "%.0f°")
            slider("Pitch", $pitchDeg, pitchRange, "%.0f°")
            slider("FOV", $fovDeg, fovRange, "%.0f°")
        } header: {
            HStack {
                Text("Camera")
                Spacer()
                Picker("Easing", selection: $easing) {
                    ForEach(CameraEasing.allCases) { Text($0.rawValue).tag($0) }
                }
                .labelsHidden()
                .pickerStyle(.menu)
            }
            .textCase(nil)
        }
    }

    /// Matches the per-scene private `LabeledSlider` markup (label + monospaced
    /// value readout over a shift-snap `Slider`) so the look is unchanged.
    @ViewBuilder
    private func slider(_ label: String, _ value: Binding<Double>,
                        _ range: ClosedRange<Double>, _ format: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: value.shiftSnapped(format: format, in: range), in: range)
        }
    }
}
