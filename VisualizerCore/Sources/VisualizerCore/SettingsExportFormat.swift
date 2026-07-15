import SwiftUI

// Platform-neutral settings-export formatting. Lives outside
// SettingsExportSection.swift so exportText() builders (used by the renderer
// settings on every platform) don't drag the AppKit-only export UI along.
public enum SettingsExportFormat {
    public static func fmt(_ d: Double) -> String {
        String(format: "%.4f", d)
    }

    public static func color(_ c: Color) -> String {
        let comps = PlatformColor(c).srgbComponents ?? SIMD4(0, 0, 0, 1)
        return String(format: "(%.4f, %.4f, %.4f)", comps.x, comps.y, comps.z)
    }
}
