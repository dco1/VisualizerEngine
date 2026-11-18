//  Tolerance-driven sampling of a smooth 2D curve: as few points as keep every chord within a set distance of the
//  curve. Mesh3.arc / bezier / catmullRom (Mesh3+Sweep) take a fixed count, which wastes points where a curve runs
//  straight and runs short where it turns hard.

import simd

/// Points along a smooth 2D curve, spaced by how much it bends: close together where it turns, far apart where it runs
/// straight, so every chord strays from the curve by about the same small distance.
///
/// A chord of length L on a curve of curvature κ strays κL²/8 from it, so a chord that strays `tolerance` spans
/// √(8·tolerance) of ∫√κ ds. Spacing the samples evenly in that integral gives the fewest points for the tolerance, and
/// a curve symmetric in its parameter gets symmetric samples. Corners are not features here: split a curve at its
/// corners and sample each smooth piece.
public enum CurveSampling {
    /// Parameter values from `t0` to `t1` (both ends included, in order) at which to sample `curve` so no chord strays
    /// more than about `tolerance` from it. The bending is measured over `resolution` even steps, so a feature
    /// narrower than one step is missed. A straight curve gives just its two ends.
    public static func parameters(_ curve: (Double) -> Vec2, from t0: Double, to t1: Double, tolerance: Double,
                                  resolution: Int = 512, maxSegments: Int = 4096) -> [Double] {
        precondition(tolerance > 0 && resolution >= 2 && t1 > t0)
        let ts = (0...resolution).map { t0 + (t1 - t0) * Double($0) / Double(resolution) }
        let p = ts.map(curve)
        // √κ ds at each inner fine point: the turn θ over the local step ds is κ ≈ θ/ds, so √κ·ds = √(θ·ds).
        var w = [Double](repeating: 0, count: p.count)
        for i in 1..<resolution {
            let a = p[i] - p[i - 1], b = p[i + 1] - p[i]
            let turn = abs(atan2(cross2(a, b), simd_dot(a, b)))
            w[i] = (turn * (simd_length(a) + simd_length(b)) / 2).squareRoot()
        }
        w[0] = w[1]; w[resolution] = w[resolution - 1]
        var g = [0.0]                                            // ∫√κ ds up to each fine point
        g.reserveCapacity(p.count)
        for i in 0..<resolution { g.append(g[i] + (w[i] + w[i + 1]) / 2) }
        let total = g[resolution]
        let n = min(maxSegments, max(1, Int((total / (8 * tolerance).squareRoot() - 1e-9).rounded(.up))))
        guard n > 1 else { return [t0, t1] }
        var out = [t0], j = 0
        for k in 1..<n {
            let target = total * Double(k) / Double(n)
            while g[j + 1] < target { j += 1 }
            let f = (target - g[j]) / max(g[j + 1] - g[j], .leastNormalMagnitude)
            out.append(ts[j] + (ts[j + 1] - ts[j]) * f)
        }
        out.append(t1)
        return out
    }

    /// The curve's points at `parameters(curve, from:to:tolerance:)`.
    public static func points(_ curve: (Double) -> Vec2, from t0: Double, to t1: Double, tolerance: Double,
                              resolution: Int = 512, maxSegments: Int = 4096) -> [Vec2] {
        parameters(curve, from: t0, to: t1, tolerance: tolerance, resolution: resolution, maxSegments: maxSegments).map(curve)
    }
}
