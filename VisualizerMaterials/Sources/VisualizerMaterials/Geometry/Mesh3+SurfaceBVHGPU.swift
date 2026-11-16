//  Mesh3+SurfaceBVHGPU.swift — a bounding-volume hierarchy of a triangle surface, built on the CPU and queried on
//  the GPU: the closest point on the surface to any number of points (`bvh_closest` in the remesher's kernels).
//
//  Guarantees: `closest` returns, to Float precision, the exact closest point on the triangles it was built from (the
//  traversal prunes only boxes farther than the best hit found), for points anywhere, however far from the surface
//  (the CPU `SurfaceProjector` gives up after 6 grid rings). The build is a Morton-order sort of triangle centres and
//  a balanced split of that order (O(n log n), a few ms per 100k triangles); leaves hold up to 4 triangles.
//  Does not guarantee: the best possible tree quality (a median split of the Morton order, not SAH).
//  Why the build is CPU: it is one sort and a recursion over ~n/4 nodes, run once per remesh, not per iteration.

import simd

#if canImport(Metal)
import Metal

struct SurfaceBVHGPU {
    let lo: MTLBuffer, hi: MTLBuffer, meta: MTLBuffer, ta: MTLBuffer, tb: MTLBuffer, tc: MTLBuffer
    let nodeCount: Int

    /// `tris` are triangles as three float4 corners each.
    init?(_ g: ModelGPU, tris: [(SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)]) {
        guard !tris.isEmpty else { return nil }
        let n = tris.count
        var cmin = SIMD3<Float>(repeating: .infinity), cmax = SIMD3<Float>(repeating: -.infinity)
        let cen = tris.map { t -> SIMD3<Float> in
            let c = (t.0.xyz + t.1.xyz + t.2.xyz) / 3
            cmin = simd_min(cmin, c); cmax = simd_max(cmax, c)
            return c
        }
        let ext = simd_max(cmax - cmin, SIMD3<Float>(repeating: 1e-20))
        func spread(_ x: UInt32) -> UInt64 {
            var v = UInt64(x) & 0x1FFFFF
            v = (v | v << 32) & 0x1F00000000FFFF; v = (v | v << 16) & 0x1F0000FF0000FF
            v = (v | v << 8) & 0x100F00F00F00F00F; v = (v | v << 4) & 0x10C30C30C30C30C3
            return (v | v << 2) & 0x1249249249249249
        }
        let codes: [UInt64] = cen.map { c in
            let q = simd_clamp((c - cmin) / ext, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1)) * 2097151
            return spread(UInt32(q.x)) | spread(UInt32(q.y)) << 1 | spread(UInt32(q.z)) << 2
        }
        let order = (0 ..< n).sorted { codes[$0] < codes[$1] || (codes[$0] == codes[$1] && $0 < $1) }
        var tlo: [SIMD4<Float>] = [], thi: [SIMD4<Float>] = [], meta: [SIMD4<UInt32>] = []
        tlo.reserveCapacity(n / 2 + 2); thi.reserveCapacity(n / 2 + 2); meta.reserveCapacity(n / 2 + 2)
        func build(_ a: Int, _ b: Int) -> Int {            // order[a ..< b]
            let id = tlo.count
            tlo.append(.zero); thi.append(.zero); meta.append(.zero)
            if b - a <= 4 {
                var l = SIMD3<Float>(repeating: .infinity), h = SIMD3<Float>(repeating: -.infinity)
                for i in a ..< b { let t = tris[order[i]]
                    l = simd_min(l, simd_min(t.0.xyz, simd_min(t.1.xyz, t.2.xyz))); h = simd_max(h, simd_max(t.0.xyz, simd_max(t.1.xyz, t.2.xyz))) }
                tlo[id] = SIMD4(l, 0); thi[id] = SIMD4(h, 0); meta[id] = SIMD4(UInt32(a), UInt32(b - a), 1, 0)
            } else {
                let m = (a + b) / 2
                let l = build(a, m), r = build(m, b)
                tlo[id] = SIMD4(simd_min(tlo[l].xyz, tlo[r].xyz), 0); thi[id] = SIMD4(simd_max(thi[l].xyz, thi[r].xyz), 0)
                meta[id] = SIMD4(UInt32(l), UInt32(r), 0, 0)
            }
            return id
        }
        _ = build(0, n)
        lo = g.buffer(tlo); hi = g.buffer(thi); self.meta = g.buffer(meta)
        ta = g.buffer(order.map { tris[$0].0 }); tb = g.buffer(order.map { tris[$0].1 }); tc = g.buffer(order.map { tris[$0].2 })
        nodeCount = tlo.count
    }

    var buffers: [MTLBuffer] { [lo, hi, meta, ta, tb, tc] }

    /// Closest surface point to each query point (Float). nil on a GPU error.
    func closest(_ g: ModelGPU, _ pts: [SIMD4<Float>]) -> [SIMD4<Float>]? {
        guard !pts.isEmpty else { return [] }
        let pb = g.buffer(pts), out = g.buffer(count: pts.count, SIMD4<Float>.self)
        guard g.run({ s in s.go("bvh_query", pts.count, [pb, out] + buffers, GP(a: UInt32(pts.count))) }) else { return nil }
        return out.array(SIMD4<Float>.self, count: pts.count)
    }
}

extension SIMD4 where Scalar == Float { var xyz: SIMD3<Float> { SIMD3(x, y, z) } }
#endif
