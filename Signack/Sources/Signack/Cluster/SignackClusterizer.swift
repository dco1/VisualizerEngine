import simd

/// Splits a triangle list into small, spatially and topologically coherent CLUSTERS — the unit
/// Signack selects, culls and (from S2 on) dispatches.
///
/// Greedy growth, meshoptimizer-style: seed at the next unassigned atom in Morton order, then keep
/// adding the edge-adjacent unassigned atom nearest the cluster's running centre until the next one
/// would overflow `maxTriangles`. When a cluster's frontier runs dry (an island is used up, or the
/// geometry has no shared edges at all, like leaf cards) it continues with the next unassigned
/// atom in Morton order — a spatial neighbour — so clusters stay compact either way.
///
/// An ATOM is indivisible: every triangle of a leaf card lands in the same cluster, because a
/// simplifier removes cards whole (`SignackCardSimplifier`). For an ordinary mesh each triangle is
/// its own atom.
public struct SignackClusterizer: Sendable, Equatable {
    public var maxTriangles: Int

    public init(maxTriangles: Int = 128) { self.maxTriangles = max(1, maxTriangles) }

    /// Partition `triangles` (with optional per-triangle `atoms`) into clusters, each a list of
    /// indices into `triangles`. Deterministic.
    public func clusters(_ triangles: [SIMD3<UInt32>], atoms: [UInt32]?,
                         mesh: SignackMeshView) -> [[Int]] {
        guard !triangles.isEmpty else { return [] }
        // Atoms: triangle lists, in first-seen order.
        var atomTris: [[Int]] = []
        if let atoms {
            var index: [UInt32: Int] = [:]
            for (t, a) in atoms.enumerated() {
                if let i = index[a] { atomTris[i].append(t) } else { index[a] = atomTris.count; atomTris.append([t]) }
            }
        } else {
            atomTris = triangles.indices.map { [$0] }
        }
        let n = atomTris.count
        let pid = mesh.positionID
        var centre = [SIMD3<Float>](repeating: .zero, count: n)
        for (a, tris) in atomTris.enumerated() {
            var c = SIMD3<Float>(repeating: 0)
            for t in tris {
                let tri = triangles[t]
                c += mesh.positions[Int(tri.x)] + mesh.positions[Int(tri.y)] + mesh.positions[Int(tri.z)]
            }
            centre[a] = c / Float(3 * tris.count)
        }
        let adjacency = Self.adjacency(atomTris, triangles, pid)
        let order = SignackMorton.order(centre)

        var assigned = [Bool](repeating: false, count: n)
        var inFrontier = [Bool](repeating: false, count: n)
        var result: [[Int]] = []
        var cursor = 0
        while true {
            while cursor < n, assigned[order[cursor]] { cursor += 1 }
            guard cursor < n else { break }
            var cluster: [Int] = []
            var frontier: [Int] = []
            var sum = SIMD3<Float>(repeating: 0)
            var count = 0
            var next = order[cursor]
            var scan = cursor
            while true {
                // Take `next`.
                assigned[next] = true
                cluster.append(contentsOf: atomTris[next])
                sum += centre[next]; count += 1
                for b in adjacency[next] where !assigned[b] && !inFrontier[b] {
                    inFrontier[b] = true; frontier.append(b)
                }
                // Choose the following atom: nearest frontier atom, else next spatial neighbour.
                frontier.removeAll { assigned[$0] }
                let c = sum / Float(count)
                var pick = -1
                var best = Float.greatestFiniteMagnitude
                for (k, b) in frontier.enumerated() {
                    let d = simd_distance_squared(centre[b], c)
                    if d < best { best = d; pick = k }
                }
                var candidate: Int
                if pick >= 0 {
                    candidate = frontier[pick]
                } else {
                    while scan < n, assigned[order[scan]] { scan += 1 }
                    guard scan < n else { break }
                    candidate = order[scan]
                }
                if cluster.count + atomTris[candidate].count > maxTriangles { break }
                if pick >= 0 { frontier.remove(at: pick) }
                inFrontier[candidate] = false
                next = candidate
            }
            for b in frontier { inFrontier[b] = false }
            result.append(cluster)
        }
        return result
    }

    /// Atoms sharing at least one position edge.
    static func adjacency(_ atomTris: [[Int]], _ triangles: [SIMD3<UInt32>],
                          _ pid: [UInt32]) -> [[Int]] {
        var edgeAtoms: [UInt64: [Int32]] = [:]
        edgeAtoms.reserveCapacity(triangles.count * 2)
        for (a, tris) in atomTris.enumerated() {
            for t in tris {
                let (e0, e1, e2) = SignackEdges.keys(triangles[t], pid)
                for e in [e0, e1, e2] {
                    if edgeAtoms[e]?.last != Int32(a) { edgeAtoms[e, default: []].append(Int32(a)) }
                }
            }
        }
        var adj = [[Int]](repeating: [], count: atomTris.count)
        for (_, list) in edgeAtoms where list.count > 1 {
            for i in list { for j in list where j != i { adj[Int(i)].append(Int(j)) } }
        }
        for i in adj.indices where adj[i].count > 1 {
            adj[i].sort()
            var out: [Int] = []
            for x in adj[i] where out.last != x { out.append(x) }
            adj[i] = out
        }
        return adj
    }
}
