import Foundation

/// **The PATTERN auditor — "does this bake repeat inside its own tile, and which way is it
/// stretched?"**
///
/// `TextureAudit` asks whether a bake is plausible and `MaterialScaleAudit` asks how big its
/// features are. Neither asks the two questions whose answers drew corrugated ribs across every
/// brushed-metal appliance front (Daydream DH-0965):
///
///  1. **`repeatScore`** — the strongest return of the circular autocorrelation along one axis,
///     measured only AFTER the field has first decorrelated. A random field decorrelates and
///     stays near 0; a field that repeats every `P` texels comes back to ~1 at lag `P`. The
///     trap it catches is `Noise.fbmTiled(u · K, v, baseCells: C)`: the lattice wraps at `C`
///     cells, so scaling the coordinate by an integer `K` does not make finer noise — it tiles
///     the SAME `C`-cell strip `K` times, an exactly periodic stripe at `1/K` of the tile. On a
///     1 m millwork tile, aluminium's `u · 40` was a 25 mm rib. Stretched noise has to come from
///     a non-square lattice (`Noise.fbmTiledAniso`), not from a scaled coordinate.
///  2. **`elongation`** — the structure tensor of the field's gradient. Its weak eigenvector is
///     the direction the pattern is LONG in (least change); `strength` is the eigenvalue
///     contrast, 0 = isotropic, → 1 = pure streaks. For a brushed material that direction must
///     be the declared `grainTangent`, or the anisotropic highlight is stretched one way while
///     the texture is streaked the other.
///
/// GPU-free and deterministic, like the other auditors. `repeatScore` is O(n³) — audit at 256²
/// or below.
public enum PatternAudit {

    /// Below this, a field has no meaningful repeat (random value noise sits under ~0.3).
    public static let repeatCeiling = 0.5

    /// Circular autocorrelation return along `axis` (0 = U / x, 1 = V / y) for a square
    /// toroidal `field` of side `size`. 0 when the field never decorrelates within half a tile
    /// (too few cells along that axis to say anything) or is flat.
    public static func repeatScore(_ field: [Double], size n: Int, axis: Int) -> Double {
        guard n >= 8, field.count == n * n else { return 0 }
        let mean = field.reduce(0, +) / Double(field.count)
        let f = field.map { $0 - mean }
        let variance = f.reduce(0) { $0 + $1 * $1 } / Double(f.count)
        guard variance > 1e-14 else { return 0 }
        // Wait for the field to fall to half-correlation, then score how far it climbs back
        // relative to the deepest dip so far. A threshold on the dip alone (r < 0.1) is blind to
        // a repeat riding on a slower component: brushed steel's height bottomed at 0.25 and came
        // back to exactly 1.00 every 8 texels.
        var decorrelated = false, peak = 0.0, dip = 1.0
        for lag in 1...(n / 2) {
            var s = 0.0
            for y in 0..<n {
                for x in 0..<n {
                    let a = f[y * n + x]
                    let b = axis == 0 ? f[y * n + (x + lag) % n] : f[((y + lag) % n) * n + x]
                    s += a * b
                }
            }
            let r = s / Double(n * n) / variance
            dip = min(dip, r)
            if !decorrelated { decorrelated = r < 0.5 }
            else { peak = max(peak, (r - dip) / max(1 - dip, 1e-9)) }
        }
        return peak
    }

    /// The worse of the two axes over every channel a renderer reads as structure — albedo
    /// luma, roughness and height.
    public static func repeatScore(_ ch: MaterialChannels) -> (u: Double, v: Double) {
        let n = ch.size
        let luma = ch.albedo.map { 0.2126 * $0.x + 0.7152 * $0.y + 0.0722 * $0.z }
        var u = 0.0, v = 0.0
        for field in [luma, ch.roughness, ch.height] {
            u = max(u, repeatScore(field, size: n, axis: 0))
            v = max(v, repeatScore(field, size: n, axis: 1))
        }
        return (u, v)
    }

    /// Which way `field` is stretched: the unit direction (in (u, v)) of least change, and the
    /// eigenvalue contrast `(λ₁ − λ₂) / (λ₁ + λ₂)` of its gradient structure tensor.
    public static func elongation(_ field: [Double], size n: Int) -> (axis: Vec2, strength: Double) {
        guard n >= 3, field.count == n * n else { return (Vec2(1, 0), 0) }
        var g: [(Double, Double)] = []
        g.reserveCapacity(n * n)
        for y in 0..<n {
            for x in 0..<n {
                g.append(((field[y * n + (x + 1) % n] - field[y * n + (x + n - 1) % n]) / 2,
                          (field[((y + 1) % n) * n + x] - field[((y + n - 1) % n) * n + x]) / 2))
            }
        }
        return elongation(gradients: g)
    }

    /// The same, read straight off a tangent-space normal map: a normal's (x, y) tilt IS the
    /// (negated) height gradient, so a detail band that stores only normals can be measured too.
    public static func elongation(normals: [Vec3]) -> (axis: Vec2, strength: Double) {
        elongation(gradients: normals.map { ($0.x, $0.y) })
    }

    static func elongation(gradients: [(Double, Double)]) -> (axis: Vec2, strength: Double) {
        var jxx = 0.0, jxy = 0.0, jyy = 0.0
        for (gx, gy) in gradients { jxx += gx * gx; jxy += gx * gy; jyy += gy * gy }
        let tr = jxx + jyy
        guard tr > 1e-18 else { return (Vec2(1, 0), 0) }
        let disc = ((jxx - jyy) * (jxx - jyy) / 4 + jxy * jxy).squareRoot()
        let l1 = tr / 2 + disc, l2 = tr / 2 - disc
        // Eigenvector of the SMALLER eigenvalue l2: (jxy, l2 − jxx), or the axis when diagonal.
        var ax = jxy, ay = l2 - jxx
        if abs(ax) + abs(ay) < 1e-18 { (ax, ay) = jxx <= jyy ? (1, 0) : (0, 1) }
        let len = (ax * ax + ay * ay).squareRoot()
        return (Vec2(ax / len, ay / len), (l1 - l2) / (l1 + l2))
    }

    /// `|cos|` between a stretch direction and the bake's mean grain tangent — 1 = streaked
    /// exactly along the declared brush, 0 = across it. `nil` when the bake declares no grain.
    public static func grainAgreement(_ ch: MaterialChannels, axis e: Vec2) -> Double? {
        guard let g = ch.grainTangent, !g.isEmpty else { return nil }
        // Axial mean (a grain is a line, not an arrow): average the doubled angle.
        var sx = 0.0, sy = 0.0
        for t in g { let a = 2 * atan2(t.y, t.x); sx += cos(a); sy += sin(a) }
        let theta = atan2(sy, sx) / 2
        return abs(e.x * cos(theta) + e.y * sin(theta))
    }
}
