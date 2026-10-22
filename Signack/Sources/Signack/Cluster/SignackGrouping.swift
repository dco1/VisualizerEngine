import simd

/// Groups neighbouring clusters (≈ `groupSize` each) — the unit the DAG simplifies together.
///
/// The Nanite trick lives here: a group is merged and simplified as one mesh with its OUTER boundary
/// locked, then split back into clusters. Because the boundary of a group never moves, the clusters
/// produced from it fit their neighbours at every level, and a cut that mixes levels has no cracks.
/// Grouping by SHARED BOUNDARY (most shared edges first) keeps those locked edges short, so the
/// simplifier has the most freedom; clusters with no shared edges (leaf cards, islands) group with
/// their nearest Morton-order neighbours.
struct SignackGrouping {
    var groupSize: Int = 4

    /// `clusterTris[i]` = cluster i's triangles; returns groups as lists of cluster indices.
    func groups(_ clusterTris: [[SIMD3<UInt32>]], mesh: SignackMeshView) -> [[Int]] {
        let n = clusterTris.count
        guard n > 0 else { return [] }
        let pid = mesh.positionID
        // Each cluster's open edges, and which clusters share each.
        var edgeOwners: [UInt64: [Int32]] = [:]
        var centre = [SIMD3<Float>](repeating: .zero, count: n)
        for (i, tris) in clusterTris.enumerated() {
            var c = SIMD3<Float>(repeating: 0)
            for t in tris {
                c += mesh.positions[Int(t.x)] + mesh.positions[Int(t.y)] + mesh.positions[Int(t.z)]
            }
            centre[i] = c / Float(max(1, 3 * tris.count))
            for e in SignackEdges.openEdges(tris, pid) { edgeOwners[e, default: []].append(Int32(i)) }
        }
        var shared = [[Int: Int]](repeating: [:], count: n)
        for (_, owners) in edgeOwners where owners.count > 1 {
            for a in owners { for b in owners where b != a { shared[Int(a)][Int(b), default: 0] += 1 } }
        }
        let order = SignackMorton.order(centre)
        var position = [Int](repeating: 0, count: n)
        for (k, i) in order.enumerated() { position[i] = k }

        var assigned = [Bool](repeating: false, count: n)
        var result: [[Int]] = []
        var cursor = 0
        while true {
            while cursor < n, assigned[order[cursor]] { cursor += 1 }
            guard cursor < n else { break }
            var group = [order[cursor]]
            assigned[order[cursor]] = true
            while group.count < groupSize {
                // Most shared boundary with the group so far; ties to the nearest centre.
                var score: [Int: Int] = [:]
                for g in group { for (b, k) in shared[g] where !assigned[b] { score[b, default: 0] += k } }
                let gc = group.reduce(SIMD3<Float>(repeating: 0)) { $0 + centre[$1] } / Float(group.count)
                var pick = -1, bestScore = 0
                var bestDist = Float.greatestFiniteMagnitude
                for (b, s) in score.sorted(by: { $0.key < $1.key }) {
                    let d = simd_distance_squared(centre[b], gc)
                    if s > bestScore || (s == bestScore && d < bestDist) { pick = b; bestScore = s; bestDist = d }
                }
                if pick < 0 {
                    // No shared boundary: the nearest unassigned among the next few in Morton order.
                    var k = position[group[0]] + 1
                    var seen = 0
                    while k < n, seen < 8 {
                        let b = order[k]
                        if !assigned[b] {
                            let d = simd_distance_squared(centre[b], gc)
                            if d < bestDist { bestDist = d; pick = b }
                            seen += 1
                        }
                        k += 1
                    }
                }
                guard pick >= 0 else { break }
                assigned[pick] = true
                group.append(pick)
            }
            result.append(group)
        }
        return result
    }
}
