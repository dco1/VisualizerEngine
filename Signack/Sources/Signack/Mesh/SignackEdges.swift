import simd

/// Undirected edges between POSITIONS (see `SignackMeshView`), the currency of every topology
/// question in the bake: which triangles touch, where a group's boundary runs, what is open.
enum SignackEdges {
    /// Order-independent key of the edge between position ids `a` and `b`.
    @inline(__always) static func key(_ a: UInt32, _ b: UInt32) -> UInt64 {
        a < b ? (UInt64(a) << 32) | UInt64(b) : (UInt64(b) << 32) | UInt64(a)
    }

    /// The three position-edge keys of a triangle.
    @inline(__always) static func keys(_ t: SIMD3<UInt32>, _ pid: [UInt32]) -> (UInt64, UInt64, UInt64) {
        let a = pid[Int(t.x)], b = pid[Int(t.y)], c = pid[Int(t.z)]
        return (key(a, b), key(b, c), key(c, a))
    }

    /// How many of `triangles` use each position edge.
    static func valence(_ triangles: [SIMD3<UInt32>], _ pid: [UInt32]) -> [UInt64: Int32] {
        var v: [UInt64: Int32] = [:]
        v.reserveCapacity(triangles.count * 2)
        for t in triangles {
            let (e0, e1, e2) = keys(t, pid)
            v[e0, default: 0] += 1; v[e1, default: 0] += 1; v[e2, default: 0] += 1
        }
        return v
    }

    /// Edges used by exactly one triangle — the open boundary of `triangles` (by position, so a
    /// UV seam is not a boundary). Sorted, so two boundaries compare with `==`.
    static func openEdges(_ triangles: [SIMD3<UInt32>], _ pid: [UInt32]) -> [UInt64] {
        valence(triangles, pid).filter { $0.value == 1 }.map(\.key).sorted()
    }
}
