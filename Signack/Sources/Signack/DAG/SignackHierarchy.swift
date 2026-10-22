import simd

/// A baked cluster LOD DAG over ONE shared vertex buffer.
///
/// The host's vertex buffer is `0 ..< sourceVertexCount` exactly as it supplied it, followed by
/// `derivedVertices` (copies of host vertices at a new position — enlarged leaf cards; empty for
/// edge-collapsed meshes, whose every level indexes level 0's vertices). Every level of every
/// cluster indexes that one buffer, so a host registers its vertices ONCE and switches LOD by
/// switching index ranges.
public struct SignackHierarchy: Sendable {
    public let sourceVertexCount: Int
    public let derivedVertices: [SignackDerivedVertex]
    /// Every cluster's triangles, concatenated (3 per triangle).
    public let indices: [UInt32]
    public let clusters: [SignackCluster]
    public let levelCount: Int
    /// Position id per vertex (host + derived) — what crack checks compare edges by.
    public let positionID: [UInt32]

    public var vertexCount: Int { sourceVertexCount + derivedVertices.count }

    /// A sphere around the whole source mesh (the union of level 0's cluster bounds).
    public var bounds: SignackSphere {
        SignackBounds.union(clusters.filter { $0.level == 0 }.map(\.bounds))
    }

    /// Triangles in level 0 (the source mesh).
    public var sourceTriangleCount: Int { triangleCount(level: 0) }

    public func triangleCount(level: Int) -> Int {
        clusters.reduce(0) { $0 + ($1.level == level ? Int($1.triangleCount) : 0) }
    }

    /// Triangles drawn by `cut` (cluster ids).
    public func triangleCount<C: Collection>(_ cut: C) -> Int where C.Element == UInt32 {
        cut.reduce(0) { $0 + Int(clusters[Int($1)].triangleCount) }
    }

    /// The cut as ONE flat index list (cluster runs concatenated in `cut` order) — what a host
    /// uploads to draw it.
    public func flatten<C: Collection>(_ cut: C) -> [UInt32] where C.Element == UInt32 {
        var out: [UInt32] = []
        out.reserveCapacity(3 * triangleCount(cut))
        for c in cut { out.append(contentsOf: indices[clusters[Int(c)].indexRange]) }
        return out
    }

    /// Materialise the host vertex buffer this hierarchy indexes: the host's vertices, then one copy
    /// per derived vertex with its position replaced. `position` reads/writes a vertex's position.
    public func vertices<V>(host: [V], position: WritableKeyPath<V, SIMD3<Float>>) -> [V] {
        precondition(host.count == sourceVertexCount, "host vertex count does not match the bake")
        var out = host
        out.reserveCapacity(vertexCount)
        for d in derivedVertices {
            var v = host[Int(d.source)]
            v[keyPath: position] = d.position(source: host[Int(d.source)][keyPath: position])
            out.append(v)
        }
        return out
    }
}
