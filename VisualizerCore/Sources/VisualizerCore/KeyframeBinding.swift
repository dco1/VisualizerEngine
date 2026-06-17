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
}
