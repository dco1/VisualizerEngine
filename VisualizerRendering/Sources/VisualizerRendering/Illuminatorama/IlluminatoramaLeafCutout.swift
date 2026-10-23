import Foundation
import simd

/// Host mirror of `IlluminatoramaLeafCutout.h` (Daydream DH-1487): the same species leaf outlines
/// in a crown-fill card's coordinates, for the parity test and for a bake that needs a leaf's area
/// (`area(code:)` — each cutout card is scaled so its leaf covers what the old ovate card did).
public enum IlluminatoramaLeafCutout {
    /// The cutout card's outline (roomier than the ovate card: the leaf, not the card, is the shape),
    /// as (v, u) of the lower and upper margin points; base (0, 0) and tip (1, 0).
    public static let cardLower = SIMD2<Float>(0.15, 0.95), cardUpper = SIMD2<Float>(0.85, 0.95)

    static func smooth(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t)
    }

    /// The shader's `illumiLeafCutoutKeep`, line for line.
    public static func keep(_ cardUV: SIMD2<Float>) -> Bool {
        let code = Int((cardUV.y * 0.5).rounded(.down))
        if code <= 0 { return true }
        let v = cardUV.y - 2 * Float(code), u = cardUV.x, au = abs(u)
        if v <= 0 || v >= 1 { return false }
        switch code {
        case 1:
            let env = 0.90 * pow(sin(Float.pi * pow(v, 1.55)), 0.85)
            let side: Float = u > 0 ? 0.035 : -0.035
            let lobe = 0.5 + 0.5 * cos(2 * Float.pi * (v - 0.14 + side) / 0.19)
            let depth = 0.30 + 0.22 * smooth(0.2, 0.7, v)
            return au < env * (1 - depth + depth * lobe.squareRoot()) * smooth(0, 0.07, v)
        case 2:
            let x = u * 0.494, y = v - 0.48
            let r = (x * x + y * y).squareRoot(), a = abs(atan2(x, y))
            let r0: Float = 0.22, spike: Float = 0.35, tooth: Float = 0.10
            var rb = r0
            for (ang, A, w) in [(Float(0), Float(0.30), Float(0.70)), (1.12, 0.26, 0.66), (2.2, 0.11, 0.5)] {
                let d = abs(a - ang) / w
                if d < 1 {
                    let p = 1 - d
                    rb = max(rb, r0 + A * p * (1 - spike) + A * spike * pow(p, 7) + tooth * A * max(0, 1 - abs(d - 0.5) / 0.1))
                }
            }
            if a > 2.75 { rb = min(rb, r0 * (0.55 + 0.45 * (Float.pi - a) / 0.39)) }
            return r < rb
        case 3:
            let env = 0.86 * sin(Float.pi * pow(v, 1.25)) * (1 - 0.25 * v)
            let f = v * 18 - (v * 18).rounded(.down)
            return au < env * (1 - 0.08 * f)
        default:
            return au < 0.80 * sin(Float.pi * pow(v, 0.92))
        }
    }

    /// The cutout card's half-width at `v` (in half-widths).
    public static func cardHalfWidth(_ v: Float) -> Float {
        let lo = cardLower, up = cardUpper
        if v <= 0 || v >= 1 { return 0 }
        if v < lo.x { return lo.y * v / lo.x }
        if v < up.x { return lo.y + (up.y - lo.y) * (v - lo.x) / (up.x - lo.x) }
        return up.y * (1 - v) / (1 - up.x)
    }

    /// The leaf area of cutout `code` inside the cutout card, in (v, u) card units (the ovate card is
    /// 1.066). Midpoint sampling, `n` × `n`.
    public static func area(code: Int, n: Int = 600) -> Float {
        var hits = 0
        for j in 0..<n {
            let v = (Float(j) + 0.5) / Float(n)
            let hw = cardHalfWidth(v)
            for i in 0..<n {
                let u = -1 + 2 * (Float(i) + 0.5) / Float(n)
                if abs(u) < hw, keep(SIMD2(u, v + 2 * Float(code))) { hits += 1 }
            }
        }
        return Float(hits) * 2 / Float(n * n)
    }
}
