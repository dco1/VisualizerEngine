import simd

/// A uniform-grid spatial hash over plant discs — what the blue-noise spacing tests against while
/// it accepts plants, and what a host asks afterwards ("is there a plant here?" — Daydream's lawn
/// keeps its blades out from under shrubs with it). O(1) per query at any planting size.
public struct ScatterDiscIndex: Sendable {
    public let cell: Double
    private var buckets: [SIMD2<Int32>: [Int]] = [:]
    public private(set) var centers: [SIMD2<Double>] = []
    public private(set) var radii: [Double] = []
    /// Largest radius inserted — how far a query must look.
    public private(set) var maxRadius: Double = 0

    public init(cell: Double) { self.cell = max(0.05, cell) }

    /// An index over `points`' canopies scaled by `radiusScale` (e.g. 0.6: the bare ring under a shrub).
    public init(_ points: [ScatterPoint], radiusScale: Double = 1) {
        let r = points.map { $0.radius * radiusScale }
        self.init(cell: max(0.25, 2 * (r.max() ?? 0.25)))
        for (p, rr) in zip(points, r) { insert(p.position, radius: rr) }
    }

    private func key(_ p: SIMD2<Double>) -> SIMD2<Int32> {
        SIMD2(Int32(clamping: Int((p.x / cell).rounded(.down))), Int32(clamping: Int((p.y / cell).rounded(.down))))
    }

    public mutating func insert(_ p: SIMD2<Double>, radius: Double) {
        buckets[key(p), default: []].append(centers.count)
        centers.append(p); radii.append(radius)
        maxRadius = max(maxRadius, radius)
    }

    /// Any inserted disc i with |p − cᵢ| < `distance(rᵢ)` (the caller's spacing law)?
    public func conflicts(_ p: SIMD2<Double>, reach: Double, distance: (Double) -> Double) -> Bool {
        let k = key(p)
        let n = Int32(max(1, Int((reach / cell).rounded(.up))))
        for dy in -n...n {
            for dx in -n...n {
                guard let list = buckets[SIMD2(k.x &+ dx, k.y &+ dy)] else { continue }
                for i in list where simd_distance_squared(p, centers[i]) < distance(radii[i]) * distance(radii[i]) {
                    return true
                }
            }
        }
        return false
    }

    /// Does any disc cover `p`?
    public func covers(_ p: SIMD2<Double>) -> Bool {
        conflicts(p, reach: maxRadius) { $0 }
    }
}
