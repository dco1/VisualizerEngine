import SwiftUI

public extension Color {
    /// The single source of truth for the modulation accent — the purple a
    /// settings control adopts when it's being driven by audio/MIDI, matched by
    /// its lane in the timeline (issue #63).
    static let modulationAccent = Color(red: 0.62, green: 0.40, blue: 0.95)
}

/// Marks a settings control as modulatable, giving it the live "driven by
/// audio/MIDI" feedback (issue #63): when the named binding has an active
/// modulation assignment the control tints purple and shows a reveal badge that
/// opens the Timeline scrolled to that lane. When it isn't modulated the control
/// renders completely unchanged, so adoption is zero-cost for inactive settings.
///
/// Lives in `VisualizerCore` (depends only on `SceneTimeline`, never the app
/// target) so any scene package's settings view can attach it:
/// `LabeledSlider(...).modulatable("spawnRate", timeline: controller.sceneTimeline)`.
public struct ModulatableModifier: ViewModifier {
    let bindingName: String
    let timeline: SceneTimeline?

    public func body(content: Content) -> some View {
        let modulated = timeline?.modulation.isModulated(bindingName) ?? false
        content
            .tint(modulated ? Color.modulationAccent : nil)
            .overlay(alignment: .topTrailing) {
                if modulated {
                    Button {
                        timeline?.requestRevealModulation(bindingName)
                    } label: {
                        Image(systemName: "dot.radiowaves.left.and.right")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(Color.modulationAccent)
                            .padding(3)
                            .background(Color.modulationAccent.opacity(0.15), in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Driven by modulation — reveal in Timeline")
                    .offset(x: 4, y: -2)
                }
            }
    }
}

public extension View {
    /// Attach modulation feedback to a settings control. See `ModulatableModifier`.
    func modulatable(_ bindingName: String, timeline: SceneTimeline?) -> some View {
        modifier(ModulatableModifier(bindingName: bindingName, timeline: timeline))
    }
}
