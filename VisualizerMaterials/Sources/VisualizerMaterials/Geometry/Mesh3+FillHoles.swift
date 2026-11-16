//  Fill holes by boundary loop — Blender's "Fill Holes" (simple loops) for `Mesh3`.
//
//  Guarantees: every boundary loop of at most `maxEdges` edges that is a simple cycle (each of its vertices has
//  exactly one outgoing boundary edge) is closed with new triangles wound to match the surface (the new faces
//  are the loop's twins, so a closed result has every edge shared by exactly two triangles). Triangulation is the
//  minimum-total-area triangulation of the loop polygon (dynamic programming, O(n³), n ≤ maxEdges), never
//  using a diagonal that already exists as a mesh edge, so no non-manifold edge is made.
//  Does not guarantee: pretty triangles (no new vertices, so long loops give slivers; follow with
//  `isotropicRemeshed` for even tessellation), curvature continuity across the patch, or filling of loops
//  that touch themselves at a vertex (those are skipped).
//  Normals / UVs: rebuilt (UV of a patch corner = the UV the surface has at that loop vertex).
//
//  CPU: loop tracing + a small DP; not per-vertex number crunching.

import simd

public extension Mesh3 {

    func filledHoles(maxEdges: Int = 32, creaseDegrees: Double = 40) -> Mesh3 {
        var t = TriTopology(self, creaseDegrees: creaseDegrees)
        // Directed boundary edges in face order a→b with no twin.
        var out: [Int: [(to: Int, face: Int)]] = [:]
        for f in 0 ..< t.tri.count where t.alive[f] {
            for k in 0 ..< 3 {
                let a = t.tri[f][k], b = t.tri[f][(k + 1) % 3]
                if t.facesOn(a, b).count == 1 { out[a, default: []].append((b, f)) }
            }
        }
        var visited = Set<Int>()
        var filled = false
        for start in out.keys.sorted() where !visited.contains(start) {
            var loop: [Int] = [], uvs: [Vec2] = []
            var v = start, ok = true
            while true {
                guard let o = out[v], o.count == 1 else { ok = false; break }
                loop.append(v); uvs.append(t.uv(o[0].face, v)); visited.insert(v)
                v = o[0].to
                if v == start { break }
                if loop.count > maxEdges || visited.contains(v) { ok = false; break }
            }
            guard ok, loop.count >= 3, loop.count <= maxEdges else { continue }
            let r = Array(loop.reversed()), ru = Array(uvs.reversed())
            let n = r.count
            // Min-area triangulation of polygon r.
            var cost = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
            var split = [[Int]](repeating: [Int](repeating: -1, count: n), count: n)
            func usable(_ i: Int, _ j: Int) -> Bool {      // a diagonal must not already be a mesh edge
                j - i == 1 || (i == 0 && j == n - 1) || t.facesOn(r[i], r[j]).isEmpty
            }
            if n >= 3 { for len in 2 ..< n { for i in 0 ..< (n - len) {
                let j = i + len
                var best = Double.infinity, bk = -1
                for k in (i + 1) ..< j where usable(i, k) && usable(k, j) {
                    let a = 0.5 * len3(cross3(t.pos[r[k]] - t.pos[r[i]], t.pos[r[j]] - t.pos[r[i]]))
                    let c = cost[i][k] + cost[k][j] + a
                    if c < best { best = c; bk = k }
                }
                cost[i][j] = best; split[i][j] = bk
            } } }
            guard cost[0][n - 1].isFinite, split[0][n - 1] >= 0 else { continue }
            var stack = [(0, n - 1)]
            while let (i, j) = stack.popLast() {
                guard j - i >= 2 else { continue }
                let k = split[i][j]
                t.addFace(r[i], r[k], r[j], ru[i], ru[k], ru[j])
                stack.append((i, k)); stack.append((k, j))
            }
            filled = true
        }
        return filled ? t.toMesh3(creaseDegrees: creaseDegrees) : self
    }
}
