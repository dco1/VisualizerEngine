//  Post-pass of the mesh-intersection boolean: a closure audit and the repairs that are always safe.
//
//  Runs only when the boolean says it was not clean (`BooleanStats.repairIncomplete`: an unflippable sliver, or
//  `constraintFailures > 0`) or when `BooleanOptions.audit` asks for it, so the normal path pays nothing.
//
//  Repairs (by index, the result is welded): drop a duplicate triangle (same three vertices, same winding: keep
//  one) and drop a coincident pair with opposite winding (a zero-thickness fin: both go). Then every edge is
//  counted: `BooleanStats.openEdges` (used once) and `nonManifoldEdges` (used more than twice) are what is left,
//  and `repairIncomplete` stays true when either is non-zero.

import Foundation

extension BooleanResult {
    /// Audits (and repairs what is safe) in place; fills the stats' defect counts.
    mutating func audit() {
        var m = mesh
        let n = m.indices.count / 3
        var seen: [SIMD3<UInt32>: [Int]] = [:]
        var drop = Set<Int>()
        for t in 0..<n {
            let a = m.indices[t * 3], b = m.indices[t * 3 + 1], c = m.indices[t * 3 + 2]
            // canonical rotation keeps the winding: start at the smallest index
            let r: SIMD3<UInt32> = a <= b && a <= c ? SIMD3(a, b, c) : (b <= a && b <= c ? SIMD3(b, c, a) : SIMD3(c, a, b))
            let rev: SIMD3<UInt32> = SIMD3(r.x, r.z, r.y)
            if let same = seen[r], !same.isEmpty { drop.insert(t); stats.duplicatesRemoved += 1; continue }
            if let opp = seen[rev]?.last { drop.insert(t); drop.insert(opp); stats.duplicatesRemoved += 2; seen[rev] = nil; continue }
            seen[r, default: []].append(t)
        }
        if !drop.isEmpty {
            var idx: [UInt32] = []
            idx.reserveCapacity(m.indices.count)
            for t in 0..<n where !drop.contains(t) { idx.append(contentsOf: m.indices[(t * 3)..<(t * 3 + 3)]) }
            m.indices = idx
        }
        var count: [UInt64: Int] = [:]
        count.reserveCapacity(m.indices.count)
        var t = 0
        while t + 2 < m.indices.count {
            for k in 0..<3 {
                let a = m.indices[t + k], b = m.indices[t + (k + 1) % 3]
                count[UInt64(min(a, b)) << 32 | UInt64(max(a, b)), default: 0] += 1
            }
            t += 3
        }
        stats.openEdges = count.values.filter { $0 == 1 }.count
        stats.nonManifoldEdges = count.values.filter { $0 > 2 }.count
        stats.repairIncomplete = stats.openEdges + stats.nonManifoldEdges > 0
        mesh = m
    }
}
