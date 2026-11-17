//  GPU weld of the two operands' vertices into the point pool (item M6b). Same result as `BooleanJob.weld` (ids
//  are numbered by first occurrence, points within `eps` of an earlier one share its id) but the search is a
//  grid point-near-point query on the GPU (Float filter, 3e-7 of the extent); the CPU confirms the few candidate
//  pairs in double. The pool's hash is then built lazily for the region the new intersection points land in
//  (`BooleanPool.preload` / `hashRegion`).

import Foundation
import simd

#if canImport(Metal)
import Metal

extension GPU2Run {
    func weld() -> Bool {
        let mA = job.meshes[0], mB = job.meshes[1]
        let all = mA.positions + mB.positions
        let n = all.count
        guard n > 0 else { return false }
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for p in all { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        org = lo
        L = Float(max(len3(hi - lo), 1e-30))
        let m3 = 3e-7 * L
        guard let pb = g.buf(floats(all)), let qlo = g.zeros(n * 16), let qhi = g.zeros(n * 16),
              let rlo = g.zeros(n * 16), let rhi = g.zeros(n * 16) else { return false }
        guard g.run({ enc in
            g.go(enc, "pt_boxes", n, [.buf(pb), .buf(qlo), .buf(qhi), .val(SIMD2<Float>(m3, Float(n)))])
            g.go(enc, "pt_boxes", n, [.buf(pb), .buf(rlo), .buf(rhi), .val(SIMD2<Float>(0, Float(n)))])
        }) else { return false }
        let ext = simd_max(hi - lo, Vec3(repeating: Double(L) * 1e-6)) + Vec3(repeating: Double(m3) * 2)
        let d = UInt32(min(160, max(1, Int(cbrt(Double(n)) * 1.2))))
        var spec = GridP(lo: SIMD4(-m3, -m3, -m3, 0), inv: SIMD4(Float(Double(d) / ext.x), Float(Double(d) / ext.y), Float(Double(d) / ext.z), 0),
                         dims: SIMD4(d, d, d, 512))
        spec.lo = SIMD4(-m3, -m3, -m3, 0)
        guard let grid = g.makeGrids([(rlo, rhi, n)], spec: spec)?.first else { return false }
        guard let q = g.query(grid: grid, loQ: qlo, hiQ: qhi, nQ: n, mode: 3, margin: m3, tris: pb, pos: pb) else { return false }
        let pairs = g.read(q.pairs, SIMD2<UInt32>.self, q.count)
        lap("g.weldGPU")
        // CPU: confirm in double and number the survivors by first occurrence
        var ids = [Int32](repeating: -1, count: n)
        var pts: [Vec3] = []
        let eps = job.eps
        var pi = 0
        if pairs.isEmpty {                                   // nothing within the filter's reach of another vertex
            for i in 0..<n { ids[i] = Int32(i) }
            pts = all; pi = n
        } else { pts.reserveCapacity(n) }
        for i in 0..<(pairs.isEmpty ? 0 : n) {
            var found: Int32 = -1
            while pi < pairs.count && Int(pairs[pi].x) == i {
                let j = Int(pairs[pi].y)
                if found < 0 && len3(all[i] - all[j]) <= eps { found = ids[j] }
                pi += 1
            }
            if found >= 0 { ids[i] = found } else { ids[i] = Int32(pts.count); pts.append(all[i]) }
        }
        weldGrid = grid; weldIds = ids; weldOrg = org; weldL = L
        job.pool = BooleanPool(origin: job.pool.origin, eps: eps)
        job.pool.preload(pts)
        var off = 0
        for s in 0..<2 {
            let m = job.meshes[s]
            var t = 0
            while t + 2 < m.indices.count {
                let a = ids[off + Int(m.indices[t])], b = ids[off + Int(m.indices[t + 1])], c = ids[off + Int(m.indices[t + 2])]
                if a != b && b != c && a != c { job.tris[s].append(SIMD3(a, b, c)); job.srcTri[s].append(Int32(t / 3)) }
                t += 3
            }
            off += m.positions.count
        }
        job.stats.trianglesA = job.tris[0].count; job.stats.trianglesB = job.tris[1].count
        job.stats.seconds["g.weldPairs"] = Double(q.count)
        lap("g.weldCPU")
        return true
    }
}
#endif
