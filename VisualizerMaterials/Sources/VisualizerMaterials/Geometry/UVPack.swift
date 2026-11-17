import Foundation
import simd

// UVPack — island rotation and shelf packing into the unit square.
//
// Guarantees: islands never overlap by their bounding boxes (shelf rows, `margin` between boxes and
// from the edges of 0…1); the one common scale is the largest the shelves allow (binary search), so
// uniform texel density between islands is kept; `rotateToFit` turns each island to its minimum-area
// bounding rectangle (convex hull + one hull edge per candidate, exact), then lays the long side along u.
// `.normalizedPerIsland` instead gives every island the same UV area (more texels for small islands).
// Doesn't: nest an island into another's concave gap (it is a shelf packer, ~70–85% fill, not a skyline),
// or flip islands. CPU: a few hundred boxes, sorting and a loop.

public enum UVDensityMode: Sendable { case uniform, normalizedPerIsland }

public struct UVPackOptions: Sendable {
    public var margin = 0.01
    public var rotateToFit = true
    public var density = UVDensityMode.uniform
    public init(margin: Double = 0.01, rotateToFit: Bool = true, density: UVDensityMode = .uniform) {
        self.margin = margin; self.rotateToFit = rotateToFit; self.density = density
    }
}

public enum UVPacker {
    public static func pack(_ input: [UVIslandLayout], mesh: Mesh3, options: UVPackOptions) -> [UVIslandLayout] {
        guard !input.isEmpty else { return [] }
        var items = input.map { UVGeometry.moved(toOrigin: options.rotateToFit
                                                    ? UVGeometry.wideAlignedToMinimumRectangle($0) : $0) }
        if options.density == .normalizedPerIsland {
            items = items.map { l in
                let a = abs(l.uvArea(mesh))
                guard a > 1e-300 else { return l }
                let s = 1 / a.squareRoot()
                return l.mapped { $0 * s }
            }
        }
        let sizes = items.map { max($0.size, Vec2(1e-9, 1e-9)) }
        let m = options.margin
        let order = sizes.indices.sorted { sizes[$0].y != sizes[$1].y ? sizes[$0].y > sizes[$1].y : $0 < $1 }

        func place(_ s: Double) -> (origins: [Vec2], fits: Bool) {
            var o = [Vec2](repeating: .zero, count: sizes.count)
            var x = m, y = m, rowH = 0.0
            for i in order {
                let w = sizes[i].x * s, h = sizes[i].y * s
                if x + w > 1 - m + 1e-12, x > m { x = m; y += rowH + m; rowH = 0 }
                if x + w > 1 - m + 1e-12 || y + h > 1 - m + 1e-12 { return (o, false) }
                o[i] = Vec2(x, y); x += w + m; rowH = max(rowH, h)
            }
            return (o, true)
        }
        var lo = 0.0, hi = 1.0 / sizes.map { max($0.x, $0.y) }.max()!
        for _ in 0..<60 { let mid = (lo + hi) / 2; if place(mid).fits { lo = mid } else { hi = mid } }
        let s = lo, origins = place(s).origins
        return items.indices.map { i in items[i].mapped { $0 * s + origins[i] } }
    }
}

/// 2D helpers shared by unwrap and pack.
enum UVGeometry {
    static func moved(toOrigin l: UVIslandLayout) -> UVIslandLayout {
        let lo = l.bounds.min
        return l.mapped { $0 - lo }
    }

    static func rotate(_ p: Vec2, _ a: Double) -> Vec2 {
        let c = cos(a), s = sin(a)
        return Vec2(c * p.x - s * p.y, s * p.x + c * p.y)
    }

    /// Convex hull (Andrew's monotone chain), counter-clockwise.
    static func hull(_ pts: [Vec2]) -> [Vec2] {
        let p = pts.sorted { $0.x != $1.x ? $0.x < $1.x : $0.y < $1.y }
        guard p.count > 2 else { return p }
        var lower: [Vec2] = [], upper: [Vec2] = []
        for q in p {
            while lower.count >= 2, cross2(lower[lower.count - 1] - lower[lower.count - 2], q - lower[lower.count - 2]) <= 0 { lower.removeLast() }
            lower.append(q)
        }
        for q in p.reversed() {
            while upper.count >= 2, cross2(upper[upper.count - 1] - upper[upper.count - 2], q - upper[upper.count - 2]) <= 0 { upper.removeLast() }
            upper.append(q)
        }
        return Array(lower.dropLast()) + Array(upper.dropLast())
    }

    /// The rotation angle (radians) that makes the points' bounding box smallest.
    static func minimumRectangleAngle(_ pts: [Vec2]) -> Double {
        let h = hull(pts)
        guard h.count >= 2 else { return 0 }
        var best = 0.0, bestArea = Double.infinity
        for i in 0..<h.count {
            let d = h[(i + 1) % h.count] - h[i]
            guard len(d) > 1e-15 else { continue }
            let a = -atan2(d.y, d.x)
            var lo = Vec2(repeating: .infinity), hi = Vec2(repeating: -.infinity)
            for p in h { let q = rotate(p, a); lo = min(lo, q); hi = max(hi, q) }
            let area = (hi.x - lo.x) * (hi.y - lo.y)
            if area < bestArea - 1e-15 { bestArea = area; best = a }
        }
        return best
    }

    static func alignedToMinimumRectangle(_ l: UVIslandLayout) -> UVIslandLayout {
        let a = minimumRectangleAngle(l.uv)
        return l.mapped { rotate($0, a) }
    }

    /// Minimum rectangle, then a quarter turn if needed so the box is wider than tall.
    static func wideAlignedToMinimumRectangle(_ l: UVIslandLayout) -> UVIslandLayout {
        let r = alignedToMinimumRectangle(l)
        let s = r.size
        return s.y > s.x ? r.mapped { rotate($0, .pi / 2) } : r
    }
}
