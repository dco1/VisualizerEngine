import simd

/// Where nothing may grow — the house, paths, beds, placed pieces — as plain plan shapes.
///
/// A host describes its world once (Daydream: the house footprint, paving and street ribbons,
/// other garden plots, `Floor.solids`); the scatter asks one question per plant: is a canopy of
/// radius r at p clear of every shape? Shapes carry their own extra clearance so a path can keep a
/// wider berth than a stepping stone.
public struct ScatterMask: Sendable, Equatable {

    public struct Polygon: Sendable, Equatable {
        public var points: [SIMD2<Double>]
        public var clearance: Double
        public init(_ points: [SIMD2<Double>], clearance: Double = 0) {
            self.points = points; self.clearance = clearance
        }
    }

    /// A polyline with a half-width (a path, a street, a garden edge).
    public struct Ribbon: Sendable, Equatable {
        public var points: [SIMD2<Double>]
        public var halfWidth: Double
        public init(_ points: [SIMD2<Double>], halfWidth: Double) {
            self.points = points; self.halfWidth = halfWidth
        }
    }

    /// An oriented rectangle (a placed piece's footprint): centre, half extents, rotation (rad).
    public struct Box: Sendable, Equatable {
        public var center: SIMD2<Double>
        public var halfExtent: SIMD2<Double>
        public var angle: Double
        public var clearance: Double
        public init(center: SIMD2<Double>, halfExtent: SIMD2<Double>, angle: Double = 0, clearance: Double = 0) {
            self.center = center; self.halfExtent = halfExtent; self.angle = angle; self.clearance = clearance
        }
    }

    /// A round keep-out (a tree trunk another planting already grew, DH-1470).
    public struct Disc: Sendable, Equatable {
        public var center: SIMD2<Double>
        public var radius: Double
        public init(center: SIMD2<Double>, radius: Double) { self.center = center; self.radius = radius }
    }

    public var polygons: [Polygon] = []
    public var ribbons: [Ribbon] = []
    public var boxes: [Box] = []
    public var discs: [Disc] = []

    public init(polygons: [Polygon] = [], ribbons: [Ribbon] = [], boxes: [Box] = [], discs: [Disc] = []) {
        self.polygons = polygons; self.ribbons = ribbons; self.boxes = boxes; self.discs = discs
    }

    public var isEmpty: Bool { polygons.isEmpty && ribbons.isEmpty && boxes.isEmpty && discs.isEmpty }

    /// Is a disc of `radius` at `p` clear of every shape (each widened by its own clearance)?
    public func isClear(_ p: SIMD2<Double>, radius: Double) -> Bool {
        for s in polygons {
            if ScatterGeometry.contains(s.points, p) { return false }
            if ScatterGeometry.distanceToOutline(s.points, p) < radius + s.clearance { return false }
        }
        for r in ribbons where r.points.count >= 2 {
            for i in 0..<(r.points.count - 1)
            where ScatterGeometry.distance(p, segment: r.points[i], r.points[i + 1]) < radius + r.halfWidth {
                return false
            }
        }
        for d in discs where simd_distance(p, d.center) < radius + d.radius { return false }
        for b in boxes {
            // Into the box's frame; distance from a point to a rectangle.
            let c = cos(b.angle), s = sin(b.angle)
            let d = p - b.center
            let local = SIMD2(d.x * c + d.y * s, -d.x * s + d.y * c)
            let q = simd_abs(local) - b.halfExtent
            let outside = simd_length(simd_max(q, .zero))
            if outside < radius + b.clearance { return false }
        }
        return true
    }
}
