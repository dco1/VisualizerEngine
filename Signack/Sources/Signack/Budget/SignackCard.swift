import simd

/// One indivisible unit of budgeted geometry — a leaf card, a needle tuft, a petal.
///
/// Signack never removes a fraction of a card: a hero leaf is a lobed fan of ~20 triangles and a
/// part of one is a torn leaf, not a smaller leaf. A host describes each card by where it is, how
/// it faces, how much surface it covers and what it costs; the budget decides which survive.
public struct SignackCard: Sendable, Equatable {
    /// First triangle of the card's contiguous run in the host's triangle list.
    public var firstTriangle: Int
    /// Triangles in the run — the card's COST.
    public var triangleCount: Int
    /// Area-weighted centroid, in the host's object space.
    public var centroid: SIMD3<Float>
    /// Area-weighted unit normal (the card's facing; sign is irrelevant — cards are two-sided).
    public var normal: SIMD3<Float>
    /// Total surface area (m²) — the card's COVERAGE.
    public var area: Float

    public init(firstTriangle: Int, triangleCount: Int,
                centroid: SIMD3<Float>, normal: SIMD3<Float>, area: Float) {
        self.firstTriangle = firstTriangle
        self.triangleCount = triangleCount
        self.centroid = centroid
        self.normal = normal
        self.area = area
    }

    /// Coverage bought per triangle spent. A 2-triangle crown-fill card and a 20-triangle lobed
    /// hero leaf of similar size differ ~10× here, which is the whole case for spending a thin
    /// budget on the cheap ones once the lobes are too small to see.
    @inlinable public var areaPerTriangle: Float {
        triangleCount > 0 ? area / Float(triangleCount) : 0
    }
}
