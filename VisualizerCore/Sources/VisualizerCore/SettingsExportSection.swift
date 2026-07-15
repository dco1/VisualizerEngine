// AppKit-only utility — not compiled for Mac Catalyst (no NSScreen/NSPasteboard/
// NSEvent/NSGraphicsContext there). Render-path code never references this file.
#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
import SwiftUI

public struct SettingsExportSection: View {
    public let exportText: () -> String
    public let saveDefaults: (() throws -> URL)?

    #if DEBUG
    @State private var saveResult: SaveResult?
    #endif

    public init(
        exportText: @escaping () -> String,
        saveDefaults: (() throws -> URL)? = nil
    ) {
        self.exportText = exportText
        self.saveDefaults = saveDefaults
    }

    public var body: some View {
        Section("Export") {
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(exportText(), forType: .string)
            } label: {
                Label("Copy settings to clipboard", systemImage: "doc.on.clipboard")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            Text("Plain-text dump of every slider. Paste it back to the agent to lock in the current look as the new defaults.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            #if DEBUG
            if let saveDefaults {
                Button {
                    do {
                        let url = try saveDefaults()
                        saveResult = .success(url.lastPathComponent)
                    } catch {
                        saveResult = .failure(error.localizedDescription)
                    }
                } label: {
                    Label("Save current values as new defaults", systemImage: "square.and.arrow.down")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Text("Rewrites the literal defaults in the scene's Settings.swift file. **Run `git diff` afterwards** to review the change before committing — a stray click is one revert away from being undone, but only if you catch it.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if let saveResult {
                    switch saveResult {
                    case .success(let filename):
                        Label("Updated \(filename) — check `git diff`.", systemImage: "checkmark.circle")
                            .font(.callout)
                            .foregroundStyle(.green)
                    case .failure(let message):
                        Label(message, systemImage: "exclamationmark.triangle")
                            .font(.callout)
                            .foregroundStyle(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            #endif
        }
    }

    #if DEBUG
    private enum SaveResult {
        case success(String)
        case failure(String)
    }
    #endif
}
#endif // canImport(AppKit)
