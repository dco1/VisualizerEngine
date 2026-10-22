import simd

/// One group of clusters, merged, as handed to a simplifier.
public struct SignackGroupInput: Sendable {
    /// Triangles, indexing the hierarchy's vertex space (`SignackMeshView`).
    public var triangles: [SIMD3<UInt32>]
    /// Per-triangle atom (indivisible unit, e.g. a leaf card), or nil when every triangle is its own.
    public var atoms: [UInt32]?
    /// Position ids this group shares with geometry OUTSIDE it — the group boundary. A simplifier
    /// must not move or remove them (this is what keeps a mixed-level cut crack-free).
    public var lockedPositions: Set<UInt32>
    /// Triangles to aim for (half the input, by default).
    public var targetTriangles: Int
    /// Level-0 triangles, atoms and surface area this group stands for — what it replaced, which a
    /// coverage-preserving simplifier compares against.
    public var sourceTriangles: Int
    public var sourceAtoms: Int
    public var sourceArea: Float
}

/// A simplified group. Triangle corners with `SignackGroupOutput.newVertexFlag` set index
/// `newVertices` (the builder gives them global ids and welds duplicates).
public struct SignackGroupOutput: Sendable {
    public static let newVertexFlag: UInt32 = 0x8000_0000

    public var triangles: [SIMD3<UInt32>]
    public var atoms: [UInt32]?
    public var newVertices: [SignackDerivedVertex]
    /// Object-space geometric error this simplification introduced (metres for a house).
    public var error: Float

    public init(triangles: [SIMD3<UInt32>], atoms: [UInt32]?,
                newVertices: [SignackDerivedVertex] = [], error: Float) {
        self.triangles = triangles
        self.atoms = atoms
        self.newVertices = newVertices
        self.error = error
    }
}

/// How a group of clusters is made coarser — the one seam between the DAG and a kind of geometry.
///
/// - `SignackEdgeCollapse`: connected surfaces (trunks, furniture) — quadric half-edge collapse,
///   boundary locked, no new vertices.
/// - `SignackCardSimplifier`: disconnected cards (foliage) — importance-ranked card removal with
///   area compensation.
///
/// S5 specialises foliage behind this protocol. Return nil when the group cannot get coarser in a
/// way worth a level (the builder then makes its clusters roots of the DAG).
public protocol SignackSimplifier: Sendable {
    func simplify(_ group: SignackGroupInput, mesh: SignackMeshView) -> SignackGroupOutput?
}
