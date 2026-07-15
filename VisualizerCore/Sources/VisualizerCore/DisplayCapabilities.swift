// AppKit-only utility — not compiled for Mac Catalyst (no NSScreen/NSPasteboard/
// NSEvent/NSGraphicsContext there). Render-path code never references this file.
#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
import OSLog

@MainActor
public enum DisplayCapabilities {
    private static let log = Logger(subsystem: AppLog.subsystem, category: "displayCaps")

    public static var currentEDRHeadroom: CGFloat {
        guard let screen = NSScreen.main else { return 1.0 }
        return screen.maximumExtendedDynamicRangeColorComponentValue
    }

    public static var maxPotentialEDR: CGFloat {
        guard let screen = NSScreen.main else { return 1.0 }
        return screen.maximumPotentialExtendedDynamicRangeColorComponentValue
    }

    public static var supportsEDR: Bool {
        maxPotentialEDR > 1.01
    }

    public static func summary() -> String {
        guard let screen = NSScreen.main else { return "display: no NSScreen.main" }
        return """
            display: \(screen.localizedName) | \
            edrHeadroom(now)=\(String(format: "%.2f", screen.maximumExtendedDynamicRangeColorComponentValue))× | \
            edrHeadroom(max)=\(String(format: "%.2f", screen.maximumPotentialExtendedDynamicRangeColorComponentValue))× | \
            supportsEDR=\(supportsEDR)
            """
    }

    public static func logSummary() {
        log.notice("\(summary())")
    }
}
#endif // canImport(AppKit)
