import SwiftUI
#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
#endif

// ──────────────────────────────────────────────────────────────────────────
// Shift-to-snap slider bindings
//
// Holding ⇧ Shift while dragging any settings slider snaps the value to a
// coarse step so it lands cleanly on round numbers (e.g. Yaw → 0°, 90°, 180°).
// The step is 5× the slider's *displayed precision*, derived from its printf
// format string, so it works for every unit/range in the app:
//
//     "%.0f°"  → step 5      (Yaw: 0, 5, 10, … 90)
//     "%.1f m" → step 0.5
//     "%.2f"   → step 0.05   (0…1 sliders stay useful)
//
// Usage: wrap the binding handed to SwiftUI's `Slider`:
//     Slider(value: $value.shiftSnapped(format: format, in: range), in: range)
//
// Without Shift held, writes pass straight through unchanged.
//
// Mac Catalyst: there is no NSEvent to read live modifier flags from, so the
// binding compiles but never snaps (a plain passthrough). Keeping the API
// alive lets every shared SwiftUI settings section compile unchanged.
// ──────────────────────────────────────────────────────────────────────────

public extension Binding where Value == Double {
    /// A binding that snaps writes to a coarse step while ⇧ Shift is held.
    /// - Parameters:
    ///   - format: the slider's printf format (e.g. `"%.0f°"`); the snap step is
    ///     5× the precision it displays.
    ///   - range: optional bounds to clamp the snapped value into.
    func shiftSnapped(format: String, in range: ClosedRange<Double>? = nil) -> Binding<Double> {
        Binding<Double>(
            get: { self.wrappedValue },
            set: { newValue in
                self.wrappedValue = applyShiftSnap(newValue, format: format, in: range)
            }
        )
    }
}

/// Snaps `value` to the shift-snap step for `format` (clamped to `range`) when
/// Shift is held; otherwise returns it unchanged.
func applyShiftSnap(_ value: Double, format: String, in range: ClosedRange<Double>?) -> Double {
    guard shiftSnapIsActive() else { return value }
    let step = shiftSnapStep(forFormat: format)
    let snapped = (value / step).rounded() * step
    guard let range else { return snapped }
    return Swift.min(Swift.max(snapped, range.lowerBound), range.upperBound)
}

/// `true` when the Shift key is currently held — read from the live modifier
/// flags so it reflects the key state mid-drag.
func shiftSnapIsActive() -> Bool {
    #if canImport(AppKit) && !targetEnvironment(macCatalyst)
    NSEvent.modifierFlags.contains(.shift)
    #else
    false
    #endif
}

/// The shift-snap increment for a printf-style float `format`: 5× the smallest
/// increment the format displays. `"%.0f"` → 5, `"%.1f"` → 0.5, `"%.2f"` → 0.05.
/// Falls back to 5 when no `%.Nf` precision can be found.
public func shiftSnapStep(forFormat format: String) -> Double {
    let decimals = floatFormatDecimals(format) ?? 0
    return 5 * pow(10.0, Double(-decimals))
}

/// Extracts N from the first `%.Nf` token in a printf format string.
private func floatFormatDecimals(_ format: String) -> Int? {
    guard let tokenRange = format.range(of: #"%\.[0-9]+f"#, options: .regularExpression) else {
        return nil
    }
    let digits = format[tokenRange].dropFirst(2).dropLast()  // "%.2f" → "2"
    return Int(digits)
}
