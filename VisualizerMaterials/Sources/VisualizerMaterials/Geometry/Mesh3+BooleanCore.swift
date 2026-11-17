//  The cutting half of the mesh-intersection boolean: weld the inputs into one point pool, find the triangle
//  pairs that touch (BVH or GPU broadphase), intersect them, and retriangulate every cut triangle so that the
//  two meshes share the intersection curve vertex for vertex. Classification and assembly are in
//  `Mesh3+Boolean.swift`.
//
//  Why the output closes: every vertex is a pool id. A point where mesh A's edge meets mesh B's surface is ONE id,
//  used by A's cut triangles (as a boundary split on that edge — both triangles on either side of the edge get
//  it, found geometrically per undirected edge) and by B's (as a constraint end inside its face). Cut triangles
//  are triangulated with exactly those vertices (`CDT2`), no Steiner points on the curve or the boundary.

import Foundation
import simd

final class BooleanJob: @unchecked Sendable {
    let meshes: [Mesh3]
    let options: BooleanOptions
    var pool: BooleanPool
    let eps: Double
    var tris: [[SIMD3<Int32>]] = [[], []]       // pool ids per kept source triangle
    var srcTri: [[Int32]] = [[], []]            // source triangle number (into the mesh's index list / 3)
    var faces: [[SIMD3<Int32>]] = [[], []]      // output faces after cutting
    var faceSrc: [[Int32]] = [[], []]
    var cut = Set<UInt64>()                     // constrained edges: component barriers
    var stats = BooleanStats()
    var cons: [[Int32: [SIMD2<Int32>]]] = [[:], [:]]
    var splits: [[UInt64: [(Double, Int32)]]] = [[:], [:]]

    init(_ a: Mesh3, _ b: Mesh3, options: BooleanOptions) {
        meshes = [a, b]; self.options = options
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for m in [a, b] { for p in m.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) } }
        if lo.x > hi.x { lo = .zero; hi = .zero }
        eps = max(options.tolerance * len3(hi - lo), 1e-300)
        pool = BooleanPool(origin: lo - Vec3(repeating: eps * 64), eps: eps)
    }

    func positions() -> [Vec3] { pool.points }

    // MARK: weld

    func weld() {
        for s in 0..<2 {
            let m = meshes[s]
            var ids = [Int32](repeating: -1, count: m.positions.count)
            for (i, p) in m.positions.enumerated() { ids[i] = Int32(pool.add(p)) }
            var t = 0
            while t + 2 < m.indices.count {
                let a = ids[Int(m.indices[t])], b = ids[Int(m.indices[t + 1])], c = ids[Int(m.indices[t + 2])]
                if a != b && b != c && a != c { tris[s].append(SIMD3(a, b, c)); srcTri[s].append(Int32(t / 3)) }
                t += 3
            }
        }
        stats.trianglesA = tris[0].count; stats.trianglesB = tris[1].count
    }

    func triPoints(_ s: Int, _ t: Int, _ P: [Vec3]) -> (Vec3, Vec3, Vec3) {
        let v = tris[s][t]
        return (P[Int(v.x)], P[Int(v.y)], P[Int(v.z)])
    }

    func boxes(_ s: Int, _ P: [Vec3]) -> ([Vec3], [Vec3]) {
        let e = Vec3(repeating: eps)
        var lo = [Vec3](), hi = [Vec3]()
        lo.reserveCapacity(tris[s].count); hi.reserveCapacity(tris[s].count)
        for v in tris[s] {
            let a = P[Int(v.x)], b = P[Int(v.y)], c = P[Int(v.z)]
            lo.append(simd_min(a, simd_min(b, c)) - e); hi.append(simd_max(a, simd_max(b, c)) + e)
        }
        return (lo, hi)
    }

    // MARK: pairs

    func findPairs(_ P: [Vec3], _ boxA: ([Vec3], [Vec3]), _ boxB: ([Vec3], [Vec3])) -> [SIMD2<Int32>] {
        if options.useGPU, tris[0].count * tris[1].count >= 1_000_000,
           let g = BooleanGPU.shared, let pairs = g.broadphase(boxA, boxB) {
            stats.broadphaseOnGPU = true
            return pairs
        }
        let bvh = BoxBVH(lo: boxB.0, hi: boxB.1)
        let n = tris[0].count
        let chunkOut = SharedBuffer([[SIMD2<Int32>]](repeating: [], count: n / 64 + 2))
        let loA = boxA.0, hiA = boxA.1
        BooleanParallel.chunks(n, minChunk: 64) { r in
            var out: [SIMD2<Int32>] = []
            for i in r { bvh.query(loA[i], hiA[i]) { out.append(SIMD2(Int32(i), Int32($0))) } }
            let slot = r.lowerBound / 64
            chunkOut.withPointer { $0[slot].append(contentsOf: out) }
        }
        return chunkOut.items.flatMap { $0 }
    }

    // MARK: cut

    func cutAll() {
        let P0 = pool.points
        let boxA = boxes(0, P0), boxB = boxes(1, P0)
        let pairs = findPairs(P0, boxA, boxB)
        stats.candidatePairs = pairs.count

        let hits = SharedBuffer([PairHit](repeating: .none, count: pairs.count))
        let eps = self.eps
        let ta = tris[0], tb = tris[1]
        BooleanParallel.chunks(pairs.count, minChunk: 128) { r in
            for i in r {
                let pr = pairs[i]
                let a = ta[Int(pr.x)], b = tb[Int(pr.y)]
                let T = (P0[Int(a.x)], P0[Int(a.y)], P0[Int(a.z)])
                let S = (P0[Int(b.x)], P0[Int(b.y)], P0[Int(b.z)])
                let h = TriTri.intersect(T, S, eps: eps)
                hits.withPointer { $0[i] = h }
            }
        }
        func link(_ u: Int, _ v: Int, _ pr: SIMD2<Int32>) {
            guard u != v else { return }
            let seg = SIMD2(Int32(u), Int32(v))
            cons[0][pr.x, default: []].append(seg)
            cons[1][pr.y, default: []].append(seg)
        }
        for (i, h) in hits.items.enumerated() {
            switch h {
            case .none: break
            case .segment(let p, let q):
                stats.intersectingPairs += 1
                link(pool.add(p), pool.add(q), pairs[i])
            case .polygon(let poly):
                stats.intersectingPairs += 1
                let ids = poly.map { pool.add($0) }
                for k in 0..<ids.count { link(ids[k], ids[(k + 1) % ids.count], pairs[i]) }
            }
        }
        findEdgeSplits(boxA: boxA, boxB: boxB)
        for s in 0..<2 { retriangulate(s) }
    }

    /// Every pool point that lies on an edge of a mesh (and is not its endpoint) splits that edge, so the
    /// triangles on both sides of it agree on the boundary.
    func findEdgeSplits(boxA: ([Vec3], [Vec3]), boxB: ([Vec3], [Vec3])) {
        let P = pool.points
        let ovLo = simd_max(boxA.0.reduce(Vec3(repeating: .infinity), simd_min), boxB.0.reduce(Vec3(repeating: .infinity), simd_min))
        let ovHi = simd_min(boxA.1.reduce(Vec3(repeating: -.infinity), simd_max), boxB.1.reduce(Vec3(repeating: -.infinity), simd_max))
        guard ovLo.x <= ovHi.x, ovLo.y <= ovHi.y, ovLo.z <= ovHi.z else { return }
        var candidates: [Int32] = []
        for (i, p) in P.enumerated() where p.x >= ovLo.x && p.x <= ovHi.x && p.y >= ovLo.y && p.y <= ovHi.y
            && p.z >= ovLo.z && p.z <= ovHi.z { candidates.append(Int32(i)) }
        let e = Vec3(repeating: eps)
        for s in 0..<2 {
            let (lo, hi) = s == 0 ? boxA : boxB
            let bvh = BoxBVH(lo: lo, hi: hi)
            let tr = tris[s]
            var found: [UInt64: [(Double, Int32)]] = [:]
            for id in candidates {
                let p = P[Int(id)]
                bvh.query(p - e, p + e) { t in
                    let v = tr[t]
                    let corner = [v.x, v.y, v.z]
                    if corner.contains(id) { return }
                    for k in 0..<3 {
                        let ia = corner[k], ib = corner[(k + 1) % 3]
                        let a = P[Int(ia)], b = P[Int(ib)]
                        let ab = b - a
                        let tt = dot3(p - a, ab) / dot3(ab, ab)
                        guard tt > 0, tt < 1 else { continue }
                        if len3(p - (a + ab * tt)) <= self.eps {
                            let lo = min(ia, ib), hi = max(ia, ib)
                            let tl = ia == lo ? tt : 1 - tt
                            let key = UInt64(lo) << 32 | UInt64(hi)
                            if !(found[key]?.contains { $0.1 == id } ?? false) { found[key, default: []].append((tl, id)) }
                        }
                    }
                }
            }
            splits[s] = found
        }
    }

    // MARK: retriangulate

    struct CutResult {
        var tris: [SIMD3<Int32>] = []          // local indices
        var localId: [Int32] = []              // local → pool id (−1: the private centre)
        var centre = Vec3.zero
        var constrained: [SIMD2<Int32>] = []
        var failures = 0
    }

    func retriangulate(_ s: Int) {
        let P = pool.points
        let tr = tris[s]
        var work: [Int] = []
        var cleanCount = 0
        for t in 0..<tr.count {
            let v = tr[t]
            if cons[s][Int32(t)] != nil || edgeHasSplit(s, v.x, v.y) || edgeHasSplit(s, v.y, v.z) || edgeHasSplit(s, v.z, v.x) {
                work.append(t)
            } else {
                faces[s].append(v); faceSrc[s].append(srcTri[s][t]); cleanCount += 1
            }
        }
        let workList = work
        let results = SharedBuffer([CutResult](repeating: CutResult(), count: workList.count))
        let consS = cons[s], splitS = splits[s]
        BooleanParallel.chunks(workList.count, minChunk: 16) { r in
            for k in r {
                let t = workList[k]
                let res = Self.cutOne(tr[t], P, consS[Int32(t)] ?? [], splitS)
                results.withPointer { $0[k] = res }
            }
        }
        for (k, res) in results.items.enumerated() {
            let t = work[k]
            var ids = res.localId
            if let c = ids.firstIndex(of: -1) { ids[c] = Int32(pool.addUnshared(res.centre)) }
            for f in res.tris {
                let a = ids[Int(f.x)], b = ids[Int(f.y)], c = ids[Int(f.z)]
                faces[s].append(SIMD3(a, b, c)); faceSrc[s].append(srcTri[s][t])
            }
            for e in res.constrained { cut.insert(CDT2.key(Int(e.x), Int(e.y))) }
            stats.constraintFailures += res.failures
        }
        stats.cutTriangles += work.count
    }

    private func edgeHasSplit(_ s: Int, _ a: Int32, _ b: Int32) -> Bool {
        splits[s][UInt64(min(a, b)) << 32 | UInt64(max(a, b))] != nil
    }

    static func cutOne(_ v: SIMD3<Int32>, _ P: [Vec3], _ segs: [SIMD2<Int32>],
                               _ splits: [UInt64: [(Double, Int32)]]) -> CutResult {
        let corner = [v.x, v.y, v.z]
        var ring: [Int32] = []
        for k in 0..<3 {
            let ia = corner[k], ib = corner[(k + 1) % 3]
            ring.append(ia)
            let lo = min(ia, ib), hi = max(ia, ib)
            if var sp = splits[UInt64(lo) << 32 | UInt64(hi)] {
                sp.sort { $0.0 < $1.0 }
                var seq = sp.map { $0.1 }
                if ia != lo { seq.reverse() }
                ring.append(contentsOf: seq)
            }
        }
        // 2D frame: drop the dominant normal axis, mirror so the corners run counter-clockwise
        let a = P[Int(corner[0])], b = P[Int(corner[1])], c = P[Int(corner[2])]
        let n = cross3(b - a, c - a)
        let ax = abs(n.x) >= abs(n.y) && abs(n.x) >= abs(n.z) ? 0 : (abs(n.y) >= abs(n.z) ? 1 : 2)
        func proj(_ p: Vec3) -> Vec2 {
            var q: Vec2
            switch ax { case 0: q = Vec2(p.y, p.z); case 1: q = Vec2(p.z, p.x); default: q = Vec2(p.x, p.y) }
            return q
        }
        let mirror = Robust.orient2d(proj(a), proj(b), proj(c)) < 0
        func pt(_ id: Int32) -> Vec2 { let q = proj(P[Int(id)]); return mirror ? Vec2(-q.x, q.y) : q }

        var cdt = CDT2(ring: ring.map(pt))
        var local: [Int32: Int] = [:]
        var localId = ring
        for (i, id) in ring.enumerated() { local[id] = i }
        var centre = Vec3.zero
        for id in ring { centre += P[Int(id)] }
        centre /= Double(ring.count)
        var res = CutResult()
        res.centre = centre
        var dropped = Set<Int32>()
        for sg in segs { for id in [sg.x, sg.y] where local[id] == nil && !dropped.contains(id) {
            if let li = cdt.insert(pt(id)) {
                local[id] = li
                while localId.count <= li { localId.append(-2) }
                localId[li] = id
            } else { dropped.insert(id) }
        } }
        var done = Set<UInt64>()
        for sg in segs {
            guard let u = local[sg.x], let w = local[sg.y], u != w, done.insert(CDT2.key(u, w)).inserted else { continue }
            cdt.insertConstraint(u, w)
        }
        cdt.finish()
        // vertices that coincided with another (insert returned an existing index) keep that index's id
        for (id, li) in local where li < localId.count && localId[li] == -2 { localId[li] = id }
        res.tris = cdt.tv
        res.localId = localId
        res.failures = cdt.failures
        for key in cdt.constrained {
            let x = Int(key >> 32), y = Int(key & 0xFFFF_FFFF)
            if x < localId.count, y < localId.count, localId[x] >= 0, localId[y] >= 0 {
                res.constrained.append(SIMD2(localId[x], localId[y]))
            }
        }
        return res
    }
}
