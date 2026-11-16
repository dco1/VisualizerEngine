//  Sliver clean-up for the mesh-intersection boolean (`Mesh3+Boolean*.swift`), run on the welded triangle lists
//  of both operands before anything is intersected.
//
//  The problem: a triangle whose thinnest altitude is below the weld tolerance `eps` (three vertices within `eps`
//  of one line: typical after snapping a mesh to a coarse grid, or from a mesh builder that leaves a vertex a hair
//  off an edge) is neither a face nor an edge to the rest of the boolean. Its apex counts as "on" the long edge
//  (`findEdgeSplits` splits the neighbours there), the exact predicates still see it as a proper triangle, and
//  every cutter that crosses it gets a different answer from the triangle and from its neighbour. The result was
//  duplicate and zero-area faces and non-manifold edges.
//
//  The fix is topological and exact: a sliver `(x, y, z)` whose longest edge is `x–y` has `z` within `eps` of
//  that edge, so flipping `x–y` to `z–q` (q = the apex of the triangle across `x–y`) replaces the sliver and its
//  neighbour by two triangles that cover the neighbour's old area to within `eps`. The surface stays closed and
//  manifold by index; no vertex moves, nothing is added. A flip can make a new sliver, so it repeats (bounded).
//  A sliver that cannot be flipped (an open edge, or the new edge would already exist) is left alone and counted
//  in `BooleanStats.slivers - sliverFlips`; the boolean then runs as before and `BooleanStats.repairIncomplete`
//  says the result should be audited.
//
//  Cost: one pass over the triangles (an altitude test each) and nothing else when there is no sliver.

import Foundation
import simd

extension BooleanJob {
    /// Altitude threshold, in units of `eps`, under which a triangle is a sliver.
    static let sliverAltitude = 2.0

    /// True when the triangle's smallest altitude (area over longest edge) is at most `sliverAltitude * eps`.
    @inline(__always)
    static func isSliver(_ v: SIMD3<Int32>, _ P: UnsafeBufferPointer<Vec3>, _ limit: Double) -> Bool {
        let a = P[Int(v.x)], b = P[Int(v.y)], c = P[Int(v.z)]
        let ab = b - a, bc = c - b, ca = a - c
        let longest2 = max(dot3(ab, ab), max(dot3(bc, bc), dot3(ca, ca)))
        let cr = cross3(ab, -ca)
        // altitude = |cross| / longest edge  ≤  limit
        return dot3(cr, cr) <= limit * limit * longest2
    }

    func cleanSlivers() {
        let limit = Self.sliverAltitude * eps
        let P = pool.points
        for s in 0..<2 {
            var work: [Int] = []
            P.withUnsafeBufferPointer { pp in
                for (i, v) in tris[s].enumerated() where Self.isSliver(v, pp, limit) { work.append(i) }
            }
            guard !work.isEmpty else { continue }
            stats.slivers += work.count
            flipSlivers(side: s, work, P, limit)
        }
    }

    private func flipSlivers(side s: Int, _ initial: [Int], _ P: [Vec3], _ limit: Double) {
        @inline(__always) func key(_ a: Int32, _ b: Int32) -> UInt64 { UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b)) }
        var edge: [UInt64: Int32] = [:]                      // directed edge → triangle
        edge.reserveCapacity(tris[s].count * 3)
        for (t, f) in tris[s].enumerated() {
            edge[key(f.x, f.y)] = Int32(t); edge[key(f.y, f.z)] = Int32(t); edge[key(f.z, f.x)] = Int32(t)
        }
        func remove(_ f: SIMD3<Int32>) { edge[key(f.x, f.y)] = nil; edge[key(f.y, f.z)] = nil; edge[key(f.z, f.x)] = nil }
        func add(_ f: SIMD3<Int32>, _ t: Int) { edge[key(f.x, f.y)] = Int32(t); edge[key(f.y, f.z)] = Int32(t); edge[key(f.z, f.x)] = Int32(t) }
        func sliver(_ f: SIMD3<Int32>) -> Bool { P.withUnsafeBufferPointer { Self.isSliver(f, $0, limit) } }

        var stack = initial
        var budget = initial.count * 8 + 64
        var unresolved = 0
        while let t = stack.popLast() {
            let f = tris[s][t]
            guard sliver(f) else { continue }
            guard budget > 0 else { unresolved += 1; continue }
            budget -= 1
            // longest edge x→y, apex z
            let c = [f.x, f.y, f.z]
            var best = 0, bl = -1.0
            for k in 0..<3 {
                let l = len3(P[Int(c[(k + 1) % 3])] - P[Int(c[k])])
                if l > bl { bl = l; best = k }
            }
            let x = c[best], y = c[(best + 1) % 3], z = c[(best + 2) % 3]
            guard let nIdx = edge[key(y, x)], Int(nIdx) != t else { unresolved += 1; continue }
            let n = Int(nIdx), g = tris[s][n]
            let q = g.x != x && g.x != y ? g.x : (g.y != x && g.y != y ? g.y : g.z)
            // the new edge z–q must not exist yet (it would become non-manifold), and q ≠ z
            if q == z || edge[key(z, q)] != nil || edge[key(q, z)] != nil { unresolved += 1; continue }
            remove(f); remove(g)
            let t1 = SIMD3(z, x, q), t2 = SIMD3(q, y, z)
            tris[s][t] = t1; tris[s][n] = t2
            srcTri[s][t] = srcTri[s][n]                      // both now lie where the neighbour was
            add(t1, t); add(t2, n)
            stats.sliverFlips += 1
            stack.append(t); stack.append(n)
        }
        if unresolved > 0 { stats.repairIncomplete = true }
    }
}
