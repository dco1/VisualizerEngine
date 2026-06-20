import SwiftUI

// ──────────────────────────────────────────────────────────────────────────
// Shared, modulation-aware settings controls (issue #63)
//
// One `LabeledSlider` / `LabeledToggle` for the whole app, replacing the
// 46 per-scene private copies. Each control is built from a settings property
// KEY-PATH, which doubles as the link to that property's `KeyframeBinding`
// (matched by `KeyframeBinding.identity`). When the binding is modulated the
// real control tints purple and shows a reveal badge that opens the Timeline at
// the matching lane — automatically, with no per-scene wiring.
//
// The active `SceneTimeline` is read from the environment (injected once by the
// host), so call sites never thread it. When no timeline is present, or the
// property has no binding, the control renders exactly like a plain slider.
//
//   LabeledSlider("Spawn rate", controller.settings, \.spawnRate,
//                 in: 0.5...30, format: "%.1f / sec")
//   LabeledSlider("Max alive", controller.settings, \.maxAlive,
//                 in: 5...800, format: "%.0f")          // Int key-path
//   LabeledToggle("Blacklight", controller.settings, \.blacklight)
// ──────────────────────────────────────────────────────────────────────────

// MARK: - Environment

public extension EnvironmentValues {
    /// The active scene's timeline, injected by the host so modulation-aware
    /// settings controls can self-link without threading it through call sites.
    @Entry var sceneTimeline: SceneTimeline? = nil
}

// MARK: - Slider

public struct LabeledSlider<Root: AnyObject>: View {
    @Environment(\.sceneTimeline) private var timeline

    private let title: String
    private let root: Root
    private let identity: AnyKeyPath
    private let value: Binding<Double>
    private let range: ClosedRange<Double>
    private let format: String

    /// `Double` property.
    public init(_ title: String, _ root: Root, _ keyPath: ReferenceWritableKeyPath<Root, Double>,
                in range: ClosedRange<Double>, format: String) {
        self.title = title
        self.root = root
        self.identity = keyPath
        self.value = Binding(get: { root[keyPath: keyPath] },
                             set: { root[keyPath: keyPath] = $0 })
        self.range = range
        self.format = format
    }

    /// `Int` property (slider works in `Double`, rounds on write).
    public init(_ title: String, _ root: Root, _ keyPath: ReferenceWritableKeyPath<Root, Int>,
                in range: ClosedRange<Double>, format: String) {
        self.title = title
        self.root = root
        self.identity = keyPath
        self.value = Binding(get: { Double(root[keyPath: keyPath]) },
                             set: { root[keyPath: keyPath] = Int($0.rounded()) })
        self.range = range
        self.format = format
    }

    private var modulated: Bool { timeline?.isModulated(identity: identity) ?? false }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                if modulated {
                    Button { timeline?.requestRevealModulation(identity: identity) } label: {
                        Image(systemName: "dot.radiowaves.left.and.right")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.modulationAccent)
                    .help("Driven by modulation — reveal in Timeline")
                }
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .foregroundStyle(modulated ? Color.modulationAccent : .secondary)
                    .monospacedDigit()
            }
            .font(.callout)
            Slider(value: value.shiftSnapped(format: format, in: range), in: range)
                .tint(modulated ? Color.modulationAccent : nil)
        }
    }
}

// MARK: - Toggle

public struct LabeledToggle<Root: AnyObject>: View {
    @Environment(\.sceneTimeline) private var timeline

    private let title: String
    private let identity: AnyKeyPath
    private let isOn: Binding<Bool>

    public init(_ title: String, _ root: Root, _ keyPath: ReferenceWritableKeyPath<Root, Bool>) {
        self.title = title
        self.identity = keyPath
        self.isOn = Binding(get: { root[keyPath: keyPath] },
                            set: { root[keyPath: keyPath] = $0 })
    }

    private var modulated: Bool { timeline?.isModulated(identity: identity) ?? false }

    public var body: some View {
        Toggle(isOn: isOn) {
            HStack(spacing: 4) {
                Text(title)
                if modulated {
                    Button { timeline?.requestRevealModulation(identity: identity) } label: {
                        Image(systemName: "dot.radiowaves.left.and.right")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.modulationAccent)
                    .help("Driven by modulation — reveal in Timeline")
                }
            }
        }
        .tint(modulated ? Color.modulationAccent : nil)
    }
}
