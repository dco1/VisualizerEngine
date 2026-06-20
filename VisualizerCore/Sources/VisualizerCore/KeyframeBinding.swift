import Foundation

@MainActor
public struct KeyframeBinding {
    public enum Kind {
        case numerical
        case boolean
    }

    public let name: String
    public let displayName: String
    public let range: ClosedRange<Double>
    public let kind: Kind
    public let get: () -> Double
    public let set: (Double) -> Void

    /// Stable link to the settings property this binding drives, used to match a
    /// settings control to its binding EXACTLY (issue #63) — `KeyPath`s are
    /// `Equatable`, so a key-path-built control and a key-path-built binding link
    /// with zero string drift. `nil` for closure-built bindings (no purple link).
    public let identity: AnyKeyPath?

    public init(
        name: String,
        displayName: String,
        range: ClosedRange<Double>,
        kind: Kind,
        get: @escaping () -> Double,
        set: @escaping (Double) -> Void,
        identity: AnyKeyPath? = nil
    ) {
        self.name = name
        self.displayName = displayName
        self.range = range
        self.kind = kind
        self.get = get
        self.set = set
        self.identity = identity
    }

    public var defaultEasing: Easing {
        switch kind {
        case .numerical: return .easeInOut
        case .boolean:   return .linear
        }
    }

    public static func double(
        _ name: String,
        _ displayName: String,
        _ range: ClosedRange<Double>,
        get: @escaping () -> Double,
        set: @escaping (Double) -> Void
    ) -> KeyframeBinding {
        KeyframeBinding(name: name, displayName: displayName, range: range,
                        kind: .numerical, get: get, set: set)
    }

    public static func int(
        _ name: String,
        _ displayName: String,
        _ range: ClosedRange<Int>,
        get: @escaping () -> Int,
        set: @escaping (Int) -> Void
    ) -> KeyframeBinding {
        KeyframeBinding(
            name: name, displayName: displayName,
            range: Double(range.lowerBound)...Double(range.upperBound),
            kind: .numerical,
            get: { Double(get()) },
            set: { set(Int($0.rounded())) }
        )
    }

    public static func bool(
        _ name: String,
        _ displayName: String,
        get: @escaping () -> Bool,
        set: @escaping (Bool) -> Void
    ) -> KeyframeBinding {
        KeyframeBinding(
            name: name, displayName: displayName,
            range: 0...1,
            kind: .boolean,
            get: { get() ? 1.0 : 0.0 },
            set: { set($0 >= 0.5) }
        )
    }

    // ── Key-path factories (issue #63) ───────────────────────────
    // Build a binding from a settings property key-path. The key-path doubles
    // as the binding's identity, so the matching `.modSlider(...)` control links
    // to this binding exactly — the real slider turns purple when modulated.

    public static func double<Root: AnyObject>(
        _ name: String, _ displayName: String, _ range: ClosedRange<Double>,
        _ root: Root, _ keyPath: ReferenceWritableKeyPath<Root, Double>
    ) -> KeyframeBinding {
        KeyframeBinding(
            name: name, displayName: displayName, range: range, kind: .numerical,
            get: { root[keyPath: keyPath] }, set: { root[keyPath: keyPath] = $0 },
            identity: keyPath
        )
    }

    public static func int<Root: AnyObject>(
        _ name: String, _ displayName: String, _ range: ClosedRange<Int>,
        _ root: Root, _ keyPath: ReferenceWritableKeyPath<Root, Int>
    ) -> KeyframeBinding {
        KeyframeBinding(
            name: name, displayName: displayName,
            range: Double(range.lowerBound)...Double(range.upperBound), kind: .numerical,
            get: { Double(root[keyPath: keyPath]) },
            set: { root[keyPath: keyPath] = Int($0.rounded()) },
            identity: keyPath
        )
    }

    public static func bool<Root: AnyObject>(
        _ name: String, _ displayName: String,
        _ root: Root, _ keyPath: ReferenceWritableKeyPath<Root, Bool>
    ) -> KeyframeBinding {
        KeyframeBinding(
            name: name, displayName: displayName, range: 0...1, kind: .boolean,
            get: { root[keyPath: keyPath] ? 1.0 : 0.0 },
            set: { root[keyPath: keyPath] = $0 >= 0.5 },
            identity: keyPath
        )
    }
}
