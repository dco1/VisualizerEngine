import CoreGraphics
import Foundation

public enum Easing: String, CaseIterable, Codable, Hashable, Sendable {
    case linear
    case easeIn
    case easeOut
    case easeInOut
    case hold

    public var displayName: String {
        switch self {
        case .linear:    return "Linear"
        case .easeIn:    return "Ease In"
        case .easeOut:   return "Ease Out"
        case .easeInOut: return "Ease In Out"
        case .hold:      return "Hold"
        }
    }

    public var controlPoints: (cp1: CGPoint, cp2: CGPoint) {
        switch self {
        case .linear:    return (CGPoint(x: 1.0 / 3.0, y: 1.0 / 3.0),
                                 CGPoint(x: 2.0 / 3.0, y: 2.0 / 3.0))
        case .easeIn:    return (CGPoint(x: 0.42, y: 0),
                                 CGPoint(x: 1.0,  y: 1.0))
        case .easeOut:   return (CGPoint(x: 0,    y: 0),
                                 CGPoint(x: 0.58, y: 1.0))
        case .easeInOut: return (CGPoint(x: 0.42, y: 0),
                                 CGPoint(x: 0.58, y: 1.0))
        case .hold:      return (.zero, .zero)
        }
    }

    public func interpolate(progress x: Double) -> Double {
        if self == .hold { return 0 }
        if x <= 0 { return 0 }
        if x >= 1 { return 1 }
        let (cp1, cp2) = controlPoints
        let t = Easing.solveBezierT(forX: x, x1: Double(cp1.x), x2: Double(cp2.x))
        return Easing.bezierComponent(t: t, c1: Double(cp1.y), c2: Double(cp2.y))
    }

    private static func bezierComponent(t: Double, c1: Double, c2: Double) -> Double {
        let inv = 1 - t
        return 3 * inv * inv * t * c1 + 3 * inv * t * t * c2 + t * t * t
    }

    private static func bezierDerivative(t: Double, c1: Double, c2: Double) -> Double {
        let inv = 1 - t
        return 3 * inv * inv * c1 + 6 * inv * t * (c2 - c1) + 3 * t * t * (1 - c2)
    }

    private static func solveBezierT(forX x: Double, x1: Double, x2: Double) -> Double {
        var t = x
        for _ in 0..<8 {
            let dx = bezierComponent(t: t, c1: x1, c2: x2) - x
            if abs(dx) < 1e-6 { return t }
            let slope = bezierDerivative(t: t, c1: x1, c2: x2)
            if abs(slope) < 1e-6 { break }
            t = max(0, min(1, t - dx / slope))
        }
        return t
    }
}
