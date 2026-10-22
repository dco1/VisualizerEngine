import simd

/// The outer envelope of a cloud of cards — "how deep inside the crown is this leaf?".
///
/// A crown is read from outside: its silhouette and its sunlit outer shell are what a camera and
/// the sun actually see, and a leaf buried halfway to the trunk is seen only through gaps. The
/// shell is measured, not assumed: directions from the cloud's centre are binned on an octahedral
/// map, each bin remembers the farthest card in it, and a card's depth is its radius over that
/// bin's. Any crown shape works — lopsided, flat-topped, wind-swept — because the envelope is the
/// cards' own.
public struct SignackCrownShell: Sendable {
    /// Bins per side of the octahedral direction map.
    public static let resolution = 16

    public let center: SIMD3<Float>
    public let minY: Float
    public let maxY: Float
    /// Farthest card radius per direction bin, dilated and max-filtered (see `init`).
    private let binRadius: [Float]

    /// Build the shell of `cards`, weighting the centre by card area.
    public init(cards: [SignackCard]) {
        var sum = SIMD3<Float>(repeating: 0)
        var wsum: Float = 0
        var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
        for c in cards where c.centroid.x.isFinite && c.centroid.y.isFinite && c.centroid.z.isFinite {
            let w = max(c.area, 1e-8)
            sum += c.centroid * w
            wsum += w
            lo = min(lo, c.centroid.y)
            hi = max(hi, c.centroid.y)
        }
        center = wsum > 0 ? sum / wsum : .zero
        minY = wsum > 0 ? lo : 0
        maxY = wsum > 0 ? hi : 0

        let n = Self.resolution
        var r = [Float](repeating: 0, count: n * n)
        for c in cards {
            let d = c.centroid - center
            let len = simd_length(d)
            guard len.isFinite, len > 1e-6 else { continue }
            let b = Self.bin(d / len)
            r[b] = max(r[b], len)
        }
        // Dilate into empty bins (a sparse crown can leave directions unvisited) …
        for _ in 0..<n {
            var changed = false
            var next = r
            for y in 0..<n { for x in 0..<n where r[y * n + x] == 0 {
                var m: Float = 0
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let xx = x + dx, yy = y + dy
                    if xx >= 0, xx < n, yy >= 0, yy < n { m = max(m, r[yy * n + xx]) }
                }
                if m > 0 { next[y * n + x] = m; changed = true }
            } }
            r = next
            if !changed { break }
        }
        // … then a 3×3 max filter, so one stray outlier card cannot make its neighbours read as
        // buried and the shell is a smooth envelope rather than a per-bin spike field.
        var f = r
        for y in 0..<n { for x in 0..<n {
            var m: Float = 0
            for dy in -1...1 { for dx in -1...1 {
                let xx = x + dx, yy = y + dy
                if xx >= 0, xx < n, yy >= 0, yy < n { m = max(m, r[yy * n + xx]) }
            } }
            f[y * n + x] = m
        } }
        binRadius = f
    }

    /// 0 at the centre → 1 on the outer envelope (clamped).
    public func shellDepth(_ p: SIMD3<Float>) -> Float {
        let d = p - center
        let len = simd_length(d)
        guard len.isFinite, len > 1e-6 else { return 0 }
        let rmax = binRadius[Self.bin(d / len)]
        return rmax > 0 ? min(1, len / rmax) : 1
    }

    /// 0 at the lowest card → 1 at the highest: a proxy for sky/sun exposure.
    public func elevation(_ p: SIMD3<Float>) -> Float {
        let span = maxY - minY
        return span > 1e-6 ? min(1, max(0, (p.y - minY) / span)) : 0.5
    }

    /// Unit direction → octahedral bin index, with +Y as the pole.
    static func bin(_ d: SIMD3<Float>) -> Int {
        let l1 = abs(d.x) + abs(d.y) + abs(d.z)
        var u = d.x / l1, v = d.z / l1
        if d.y < 0 {
            let uu = (1 - abs(v)) * (u >= 0 ? 1 : -1)
            let vv = (1 - abs(u)) * (v >= 0 ? 1 : -1)
            u = uu; v = vv
        }
        let n = resolution
        let x = min(n - 1, max(0, Int((u * 0.5 + 0.5) * Float(n))))
        let y = min(n - 1, max(0, Int((v * 0.5 + 0.5) * Float(n))))
        return y * n + x
    }
}
