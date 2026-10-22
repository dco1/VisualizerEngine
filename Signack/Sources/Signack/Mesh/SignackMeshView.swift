import simd

/// What the bake knows about the hierarchy's vertex space while it builds — read-only to the
/// clusterizer and the simplifiers.
///
/// Two kinds of identity live side by side:
/// - the VERTEX (an attribute vertex — position + normal + uv + … in the host's buffer), which is
///   what triangles index, and
/// - the POSITION (`positionID`): vertices that sit at bit-identical positions share one. A UV or
///   normal seam is two vertices at one position. Topology — adjacency, boundaries, what may
///   collapse — is a question about positions; output triangles still index vertices.
///
/// Vertices `0 ..< sourceCount` are the host's. Vertices past that are DERIVED by a simplifier (a
/// leaf card enlarged about its centre for coverage); `origin[v]` is the host vertex each copies
/// every attribute from but its position.
public struct SignackMeshView: Sendable {
    public internal(set) var positions: [SIMD3<Float>]
    public internal(set) var positionID: [UInt32]
    /// Per position id: more than one vertex sits there (an attribute seam).
    public internal(set) var seam: [Bool]
    /// Per vertex: the host vertex it copies (itself for host vertices).
    public internal(set) var origin: [UInt32]
    public let sourceCount: Int

    init(positions: [SIMD3<Float>]) {
        self.positions = positions
        sourceCount = positions.count
        origin = (0..<UInt32(positions.count)).map { $0 }
        var ids: [SIMD3<UInt32>: UInt32] = [:]
        ids.reserveCapacity(positions.count)
        var pid = [UInt32](repeating: 0, count: positions.count)
        var counts: [UInt32] = []
        for (i, p) in positions.enumerated() {
            let key = SIMD3(p.x.bitPattern, p.y.bitPattern, p.z.bitPattern)
            if let j = ids[key] { pid[i] = j; counts[Int(j)] += 1 } else {
                let j = UInt32(counts.count)
                ids[key] = j
                pid[i] = j
                counts.append(1)
            }
        }
        positionID = pid
        seam = counts.map { $0 > 1 }
    }

    /// Append a derived vertex; it gets a fresh position id (derived geometry is never welded to
    /// anything — it belongs to one card).
    mutating func appendDerived(_ d: SignackDerivedVertex) -> UInt32 {
        let v = UInt32(positions.count)
        positions.append(d.position(source: positions[Int(d.source)]))
        positionID.append(UInt32(seam.count))
        seam.append(false)
        origin.append(d.source)
        return v
    }
}

/// A vertex the bake made by moving a host vertex — every attribute copied from `source`, the
/// position scaled about `center` by `scale` (a coverage-compensated leaf card).
public struct SignackDerivedVertex: Sendable, Hashable {
    public var source: UInt32
    public var center: SIMD3<Float>
    public var scale: Float

    public init(source: UInt32, center: SIMD3<Float>, scale: Float) {
        self.source = source
        self.center = center
        self.scale = scale
    }

    @inlinable public func position(source p: SIMD3<Float>) -> SIMD3<Float> {
        center + (p - center) * scale
    }
}
