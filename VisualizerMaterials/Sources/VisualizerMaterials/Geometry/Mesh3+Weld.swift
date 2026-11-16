//  Merge-by-distance — Blender's "Merge by Distance" / Remove Doubles for `Mesh3`.
//
//  Guarantees: vertices closer than `distance` (and, optionally, agreeing in UV and normal direction) become
//  one vertex, so triangles that shared a position but not an index now share the index and the mesh is
//  topologically connected. Triangles that collapse to a line or point are dropped. Winding is untouched.
//  Does NOT guarantee: that the result is manifold (a soup with a real T-junction stays open), or transitive
//  merging (a chain of points each `distance` apart is merged to the first representative only).
//
//  Attribute policy ("sensibly"):
//   · UVs: two vertices merge only when their UVs agree within `uvTolerance`, so texture seams survive.
//   · Normals: merge only when the angle between them is ≤ `normalAngleDegrees`. The default (180) merges every
//     coincident vertex and averages the normals (a closed, smooth-shaded result); pass e.g. 30 to keep hard
//     edges split (a flat-shaded cube then stays 24 vertices, still welded where faces are coplanar).
//   · Position of a merged vertex is its first (representative) vertex's, so welding never drifts geometry.
//
//  CPU: a uniform grid hash (cell = distance), O(n). This is a one-off topology pass, not per-frame work.

import simd

public extension Mesh3 {

    /// Merge vertices within `distance`. See the file header for the attribute policy.
    func welded(distance: Double, uvTolerance: Double = 1e-6, normalAngleDegrees: Double = 180) -> Mesh3 {
        guard !indices.isEmpty else { return self }
        let hasUV = uvs.count == positions.count, hasN = normals.count == positions.count
        let cosN = cos(normalAngleDegrees * .pi / 180)
        let rep = Mesh3.weldClusters(positions, distance: distance) { r, i in
            if hasUV {
                let d = uvs[r] - uvs[i]
                if abs(d.x) > uvTolerance || abs(d.y) > uvTolerance { return false }
            }
            if hasN, normalAngleDegrees < 180 { return dot3(normals[r], normals[i]) >= cosN - 1e-12 }
            return true
        }
        var newIndex = [Int](repeating: -1, count: positions.count)
        var out = Mesh3()
        var nSum: [Vec3] = []
        for i in 0 ..< positions.count where rep[i] == i {
            newIndex[i] = out.positions.count
            out.positions.append(positions[i])
            if hasUV { out.uvs.append(uvs[i]) }
            nSum.append(.zero)
        }
        for i in 0 ..< positions.count { newIndex[i] = newIndex[rep[i]]; if hasN { nSum[newIndex[i]] += normals[i] } }
        if hasN {
            let repNormal = (0 ..< positions.count).filter { rep[$0] == $0 }.map { normals[$0] }
            out.normals = nSum.enumerated().map { k, n in len3(n) > 1e-12 ? normalize3(n) : repNormal[k] }
        }
        var t = 0
        while t + 2 < indices.count {
            let a = newIndex[Int(indices[t])], b = newIndex[Int(indices[t + 1])], c = newIndex[Int(indices[t + 2])]
            if a != b, b != c, a != c {
                out.indices.append(UInt32(a)); out.indices.append(UInt32(b)); out.indices.append(UInt32(c))
            }
            t += 3
        }
        return out
    }

    /// Cluster map: `result[i]` is the representative vertex index of `i` (itself when it starts a cluster).
    /// `accept(rep, i)` can veto a merge (attributes). Grid hash with cell = `distance`, 27-cell lookup.
    static func weldClusters(_ pts: [Vec3], distance: Double, accept: (Int, Int) -> Bool = { _, _ in true }) -> [Int] {
        let cell = Swift.max(distance, 1e-12)
        func key(_ p: Vec3) -> SIMD3<Int64> {
            SIMD3(Int64((p.x / cell).rounded(.down)), Int64((p.y / cell).rounded(.down)), Int64((p.z / cell).rounded(.down)))
        }
        var grid: [SIMD3<Int64>: [Int]] = [:]
        var rep = [Int](repeating: 0, count: pts.count)
        let d2 = distance * distance
        for i in 0 ..< pts.count {
            let p = pts[i], k = key(p)
            var found = -1
            search: for dx in -1 ... 1 { for dy in -1 ... 1 { for dz in -1 ... 1 {
                guard let reps = grid[SIMD3(k.x + Int64(dx), k.y + Int64(dy), k.z + Int64(dz))] else { continue }
                for r in reps {
                    let q = pts[r] - p
                    if dot3(q, q) <= d2, accept(r, i) { found = r; break search }
                }
            } } }
            if found >= 0 { rep[i] = found } else { rep[i] = i; grid[k, default: []].append(i) }
        }
        return rep
    }
}
