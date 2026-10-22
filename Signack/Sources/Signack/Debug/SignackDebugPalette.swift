import Foundation
import simd

/// Distinct debug colours, so a cut is visible: one per CLUSTER (golden-ratio hue walk — neighbours
/// never match) or one per LEVEL (a fixed neon ramp, level 0 first).
public enum SignackDebugPalette {
    /// Fixed neon per level: lime, cyan, magenta, orange, yellow, violet, red, blue, then repeats.
    public static let levels: [SIMD3<Float>] = [
        SIMD3(0.45, 1.00, 0.10), SIMD3(0.05, 0.85, 1.00), SIMD3(1.00, 0.10, 0.85),
        SIMD3(1.00, 0.50, 0.05), SIMD3(1.00, 0.95, 0.10), SIMD3(0.55, 0.25, 1.00),
        SIMD3(1.00, 0.12, 0.12), SIMD3(0.12, 0.30, 1.00),
    ]

    public static func level(_ l: Int) -> SIMD3<Float> { levels[((l % levels.count) + levels.count) % levels.count] }

    /// A saturated colour for cluster `i`.
    public static func cluster(_ i: Int) -> SIMD3<Float> {
        let h = (Double(i) * 0.618_033_988_75).truncatingRemainder(dividingBy: 1)
        let v: Float = i % 3 == 0 ? 1 : (i % 3 == 1 ? 0.8 : 0.62)
        return hsv(Float(h), 0.9, v)
    }

    static func hsv(_ h: Float, _ s: Float, _ v: Float) -> SIMD3<Float> {
        let k = SIMD3<Float>(5, 3, 1)
        let q = k + h * 6
        let p = q - 6 * (q / 6).rounded(.down)
        let t = simd_clamp(simd_min(p, 4 - p), SIMD3(repeating: 0), SIMD3(repeating: 1))
        return v - v * s * t
    }
}
