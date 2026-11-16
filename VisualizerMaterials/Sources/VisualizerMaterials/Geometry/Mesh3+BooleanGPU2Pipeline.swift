//  The GPU-heavy boolean pipeline, cutting half (item M6b). Same stages and same outputs as `BooleanJob.cutAll`
//  (the CPU path stays the oracle), with these stages moved to the GPU:
//
//    broadphase      uniform grid (counting sort), one thread per A triangle, pairs in deterministic order
//    narrow phase    Float triangle–triangle filter: miss / hit (edge descriptors, endpoints emitted to a GPU
//                    buffer) / uncertain; only hits and uncertain pairs come back (stream compaction)
//    edge splits     every pool point against every triangle edge, Float filter on the GPU
//    cut triangles   which triangles carry a constraint or a split edge, and the clean ones compacted
//
//  What stays on the CPU, and why: the exact predicates and the intersection POINTS in double (the GPU has no
//  double: it names the two edges whose crossings bound a segment, the CPU evaluates them), the welded point
//  pool, and the constrained Delaunay retriangulation of the cut triangles (branchy, exact arithmetic).

import Foundation
import simd

#if canImport(Metal)
import Metal

/// `a.map(f)` on all cores (the conversions between the CPU's doubles and the GPU's floats are the CPU's share).
func pmap<A, B>(_ a: [A], _ f: @escaping @Sendable (A) -> B) -> [B] {
    let n = a.count
    return [B](unsafeUninitializedCapacity: n) { buf, count in
        nonisolated(unsafe) let out = buf.baseAddress
        a.withUnsafeBufferPointer { ap in
            nonisolated(unsafe) let inp = ap.baseAddress
            BooleanParallel.chunks(n, minChunk: 16_384) { r in
                for i in r { (out! + i).initialize(to: f(inp![i])) }
            }
        }
        count = n
    }
}

final class GPU2Run {
    let job: BooleanJob
    let g: BooleanGPU2
    var org = Vec3.zero
    var L: Float = 1
    var posBuf: MTLBuffer!
    var triBuf: [MTLBuffer] = []
    var boxLo: [MTLBuffer] = [], boxHi: [MTLBuffer] = []
    var grids: [BoxGrid2] = []
    var bounds: [(lo: Vec3, hi: Vec3)] = []
    var failed = false
    /// The weld's grid over the input vertices, and each vertex's pool id: new intersection points are looked up in it.
    var weldGrid: BoxGrid2?, weldIds: [Int32] = [], weldOrg = Vec3.zero, weldL: Float = 1
    /// The two segment endpoints (float, relative to `org`) the narrow phase emitted per pair, kept for tests.
    var endpoints: MTLBuffer?
    /// Test hooks: every candidate pair and the narrow phase's verdict on it.
    var lastPairs: MTLBuffer?, lastCode: MTLBuffer?
    var keepNarrow = false
    private var clock = Date()

    init(job: BooleanJob, g: BooleanGPU2) { self.job = job; self.g = g }

    func lap(_ name: String) {
        let now = Date()
        job.stats.seconds[name, default: 0] += now.timeIntervalSince(clock)
        clock = now
    }

    // MARK: helpers

    func floats(_ P: [Vec3]) -> [SIMD4<Float>] {
        let o = org
        return pmap(P) { SIMD4(Float($0.x - o.x), Float($0.y - o.y), Float($0.z - o.z), 0) }
    }
    var margin: Float { 2e-6 * L }

    func prepare() -> Bool {
        let P = job.pool.points
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for p in P { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        if P.isEmpty { return false }
        org = lo
        L = Float(max(len3(hi - lo), 1e-30))
        bounds = []
        for s in 0..<2 {
            var l = Vec3(repeating: .infinity), h = Vec3(repeating: -.infinity)
            for v in job.meshes[s].positions { l = simd_min(l, v); h = simd_max(h, v) }
            bounds.append((l, h))
        }
        guard let pb = g.buf(floats(P)), let t0 = g.buf(job.tris[0]), let t1 = g.buf(job.tris[1]) else { return false }
        posBuf = pb; triBuf = [t0, t1]
        return true
    }

    func gridSpec() -> GridP {
        let P = job.pool.points
        let m = Double(margin) * 2
        var rlo = simd_max(bounds[0].lo, bounds[1].lo) - Vec3(repeating: m)
        var rhi = simd_min(bounds[0].hi, bounds[1].hi) + Vec3(repeating: m)
        if rlo.x > rhi.x || rlo.y > rhi.y || rlo.z > rhi.z { rlo = bounds[0].lo; rhi = rlo + Vec3(repeating: m) }
        let ext = simd_max(rhi - rlo, Vec3(repeating: Double(L) * 1e-6))
        var mean = Vec3.zero
        var cnt = 0.0
        for s in 0..<2 {
            let tr = job.tris[s]
            let stride = max(1, tr.count / 2048)
            var i = 0
            while i < tr.count {
                let v = tr[i]
                let a = P[Int(v.x)], b = P[Int(v.y)], c = P[Int(v.z)]
                mean += simd_max(a, simd_max(b, c)) - simd_min(a, simd_min(b, c))
                cnt += 1
                i += stride
            }
        }
        mean /= max(cnt, 1)
        var dims = SIMD3<UInt32>(1, 1, 1)
        var inv = SIMD3<Float>(1, 1, 1)
        for k in 0..<3 {
            let want = max(2 * mean[k], ext[k] / 128, Double(L) * 1e-6)
            let d = min(128, max(1, Int((ext[k] / want).rounded(.up))))
            dims[k] = UInt32(d)
            inv[k] = Float(Double(d) / ext[k])
        }
        let l = rlo - org
        return GridP(lo: SIMD4(Float(l.x), Float(l.y), Float(l.z), 0), inv: SIMD4(inv.x, inv.y, inv.z, 0),
                     dims: SIMD4(dims.x, dims.y, dims.z, 512))
    }

    // MARK: cut

    func cutAll() -> Bool {
        lap("g.prep0")
        guard prepare() else { return false }
        lap("g.prepare")
        var sets: [(lo: MTLBuffer, hi: MTLBuffer, n: Int)] = []
        for s in 0..<2 {
            let n = job.tris[s].count
            guard let lo = g.zeros(n * 16), let hi = g.zeros(n * 16) else { return false }
            boxLo.append(lo); boxHi.append(hi)
            sets.append((lo, hi, n))
        }
        guard g.run({ enc in
            for s in 0..<2 {
                g.go(enc, "tri_boxes", sets[s].n, [.buf(triBuf[s]), .buf(posBuf), .buf(boxLo[s]), .buf(boxHi[s]),
                                                    .val(SIMD2<Float>(margin, Float(sets[s].n)))])
            }
        }) else { return false }
        lap("g.boxes")
        guard let gs = g.makeGrids(sets, spec: gridSpec()) else { return false }
        grids = gs
        lap("g.grid")
        guard let q = g.query(grid: grids[1], loQ: boxLo[0], hiQ: boxHi[0], nQ: job.tris[0].count, mode: 0, margin: 0,
                              tris: triBuf[1], pos: posBuf) else { return false }
        job.stats.candidatePairs = q.count
        job.stats.broadphaseOnGPU = true
        lap("g.broadphase")
        guard narrowAndLink(q.pairs, q.count) else { return false }
        guard splitEdges() else { return false }
        for s in 0..<2 { guard retriangulate(s) else { return false } }
        lap("g.retriangulate")
        return true
    }

    private struct NP { var n: UInt32; var L: Float; var p0: Float = 0; var p1: Float = 0 }

    private func narrowAndLink(_ pairBuf: MTLBuffer, _ np: Int) -> Bool {
        guard np > 0 else { return true }
        guard let code = g.zeros(np * 16), let ep = g.zeros(np * 32), let flags = g.zeros((np + 1) * 4) else { return false }
        endpoints = ep
        if keepNarrow { lastPairs = pairBuf; lastCode = code }
        guard g.run({ enc in
            g.go(enc, "narrow", np, [.buf(pairBuf), .buf(triBuf[0]), .buf(triBuf[1]), .buf(posBuf), .val(NP(n: UInt32(np), L: L)),
                                      .buf(code), .buf(ep)])
            g.go(enc, "nonmiss_flags", np, [.buf(code), .buf(flags), .val(UInt32(np))])
            g.scan(enc, flags, np + 1)
        }) else { return false }
        let nm = g.u32(flags, np)
        guard let oc = g.zeros(nm * 16), let oi = g.zeros(nm * 4) else { return false }
        guard g.run({ enc in
            g.go(enc, "compact_nonmiss", np, [.buf(code), .buf(flags), .buf(oc), .buf(oi), .val(UInt32(np))])
        }) else { return false }
        let codes = g.read(oc, SIMD4<UInt32>.self, nm), idx = g.read(oi, UInt32.self, nm)
        let pairs = g.read(pairBuf, SIMD2<UInt32>.self, np)
        lap("g.narrow")

        let P0 = job.pool.points, ta = job.tris[0], tb = job.tris[1], eps = job.eps
        let hits = SharedBuffer([PairHit](repeating: .none, count: nm))
        var nHit = 0
        for c in codes where c.x == 1 { nHit += 1 }
        for c in codes where c.x == 2 { job.stats.seconds["g.why\(c.w)", default: 0] += 1 }
        BooleanParallel.chunks(nm, minChunk: 128) { r in
            for k in r {
                let pr = pairs[Int(idx[k])]
                let a = ta[Int(pr.x)], b = tb[Int(pr.y)]
                let T = (P0[Int(a.x)], P0[Int(a.y)], P0[Int(a.z)])
                let S = (P0[Int(b.x)], P0[Int(b.y)], P0[Int(b.z)])
                let h: PairHit
                if codes[k].x == 1 {
                    h = .segment(GPU2Run.endpoint(T, S, codes[k].y), GPU2Run.endpoint(T, S, codes[k].z))
                } else {
                    h = TriTri.intersect(T, S, eps: eps)
                }
                hits.withPointer { $0[k] = h }
            }
        }
        job.stats.seconds["g.hitsDecidedOnGPU"] = Double(nHit)
        job.stats.seconds["g.uncertainToCPU"] = Double(nm - nHit)
        lap("g.resolve")
        func link(_ u: Int, _ v: Int, _ pr: SIMD2<UInt32>) {
            guard u != v else { return }
            let seg = SIMD2(Int32(u), Int32(v))
            job.cons[0][Int32(pr.x), default: []].append(seg)
            job.cons[1][Int32(pr.y), default: []].append(seg)
        }
        var hl = Vec3(repeating: .infinity), hh = Vec3(repeating: -.infinity)
        for h in hits.items {
            switch h {
            case .none: break
            case .segment(let p, let q): hl = simd_min(hl, simd_min(p, q)); hh = simd_max(hh, simd_max(p, q))
            case .polygon(let poly): for p in poly { hl = simd_min(hl, p); hh = simd_max(hh, p) }
            }
        }
        if let wg = weldGrid, hl.x <= hh.x {
            var newPts: [Vec3] = []
            for h in hits.items {
                switch h {
                case .none: break
                case .segment(let p, let q): newPts.append(p); newPts.append(q)
                case .polygon(let poly): newPts += poly
                }
            }
            let m3 = 3e-7 * weldL, wo = weldOrg
            let f4 = pmap(newPts) { SIMD4<Float>(Float($0.x - wo.x), Float($0.y - wo.y), Float($0.z - wo.z), 0) }
            let lo = f4.map { SIMD4<Float>($0.x - m3, $0.y - m3, $0.z - m3, 0) }, hi = f4.map { SIMD4<Float>($0.x + m3, $0.y + m3, $0.z + m3, 0) }
            guard let lb = g.buf(lo), let hb = g.buf(hi),
                  let q = g.query(grid: wg, loQ: lb, hiQ: hb, nQ: newPts.count, mode: 4, margin: m3, tris: lb, pos: lb) else { return false }
            let cand = g.read(q.pairs, SIMD2<UInt32>.self, q.count)
            job.pool.hashPreloaded(Array(Set(cand.map { Int(weldIds[Int($0.y)]) })).sorted())
            job.stats.seconds["g.newPointCandidates"] = Double(cand.count)
        }
        for (k, h) in hits.items.enumerated() {
            let pr = pairs[Int(idx[k])]
            switch h {
            case .none: break
            case .segment(let p, let q):
                job.stats.intersectingPairs += 1
                link(job.pool.add(p), job.pool.add(q), pr)
            case .polygon(let poly):
                job.stats.intersectingPairs += 1
                let ids = poly.map { job.pool.add($0) }
                for k in 0..<ids.count { link(ids[k], ids[(k + 1) % ids.count], pr) }
            }
        }
        lap("g.link")
        return true
    }

    /// The double-precision point named by an endpoint descriptor: the edge `k → k+1` of the S triangle (owner 0)
    /// crossing T's plane, or of T (owner 1) crossing S's plane.
    static func endpoint(_ T: (Vec3, Vec3, Vec3), _ S: (Vec3, Vec3, Vec3), _ code: UInt32) -> Vec3 {
        let owner = code & 1, k = Int(code >> 1) & 3
        let (V, W) = owner == 0 ? (S, T) : (T, S)
        let vv = [V.0, V.1, V.2]
        let n = cross3(W.1 - W.0, W.2 - W.0)
        let i = k, j = (k + 1) % 3
        let di = dot3(n, vv[i] - W.0), dj = dot3(n, vv[j] - W.0)
        let t = di == dj ? 0.5 : di / (di - dj)
        return vv[i] + (vv[j] - vv[i]) * t
    }

    // MARK: edge splits

    private func splitEdges() -> Bool {
        let P = job.pool.points
        guard let pb = g.buf(floats(P)) else { return false }
        posBuf = pb
        let e = Vec3(repeating: job.eps)
        let ovLo = simd_max(bounds[0].lo, bounds[1].lo) - e, ovHi = simd_min(bounds[0].hi, bounds[1].hi) + e
        guard ovLo.x <= ovHi.x, ovLo.y <= ovHi.y, ovLo.z <= ovHi.z else { return true }
        guard let qlo = g.zeros(P.count * 16), let qhi = g.zeros(P.count * 16) else { return false }
        guard g.run({ enc in
            g.go(enc, "pt_boxes", P.count, [.buf(posBuf), .buf(qlo), .buf(qhi), .val(SIMD2<Float>(margin, Float(P.count)))])
        }) else { return false }
        for s in 0..<2 {
            guard let q = g.query(grid: grids[s], loQ: qlo, hiQ: qhi, nQ: P.count, mode: 1, margin: 1e-6 * L,
                                  tris: triBuf[s], pos: posBuf) else { return false }
            let pairs = g.read(q.pairs, SIMD2<UInt32>.self, q.count)
            let tr = job.tris[s]
            var found: [UInt64: [(Double, Int32)]] = [:]
            let eps = job.eps
            for pr in pairs {
                let id = Int32(pr.x)
                let p = P[Int(id)]
                guard p.x >= ovLo.x, p.x <= ovHi.x, p.y >= ovLo.y, p.y <= ovHi.y, p.z >= ovLo.z, p.z <= ovHi.z else { continue }
                let v = tr[Int(pr.y)]
                let corner = [v.x, v.y, v.z]
                if corner.contains(id) { continue }
                for k in 0..<3 {
                    let ia = corner[k], ib = corner[(k + 1) % 3]
                    let a = P[Int(ia)], b = P[Int(ib)]
                    let ab = b - a
                    let tt = dot3(p - a, ab) / dot3(ab, ab)
                    guard tt > 0, tt < 1 else { continue }
                    if len3(p - (a + ab * tt)) <= eps {
                        let lo = min(ia, ib), hi = max(ia, ib)
                        let tl = ia == lo ? tt : 1 - tt
                        let key = UInt64(lo) << 32 | UInt64(hi)
                        if !(found[key]?.contains { $0.1 == id } ?? false) { found[key, default: []].append((tl, id)) }
                    }
                }
            }
            job.splits[s] = found
        }
        lap("g.splits")
        return true
    }

    // MARK: retriangulate (which triangles are cut: on the GPU)

    private func retriangulate(_ s: Int) -> Bool {
        let P = job.pool.points
        let tr = job.tris[s]
        let n = tr.count
        guard n > 0 else { return true }
        guard let consFlag = g.zeros(n * 4), let flags = g.zeros((n + 1) * 4), let srcBuf = g.buf(job.srcTri[s]) else { return false }
        let cp = consFlag.contents().assumingMemoryBound(to: UInt32.self)
        for key in job.cons[s].keys { cp[Int(key)] = 1 }
        let keys = job.splits[s].keys.sorted().map { SIMD2<UInt32>(UInt32($0 >> 32), UInt32($0 & 0xFFFF_FFFF)) }
        guard let keyBuf = g.buf(keys) else { return false }
        guard g.run({ enc in
            g.go(enc, "tri_flags", n, [.buf(triBuf[s]), .buf(consFlag), .buf(keyBuf), .val(SIMD2<UInt32>(UInt32(n), UInt32(keys.count))), .buf(flags)])
            g.scan(enc, flags, n + 1)
        }) else { return false }
        let nWork = g.u32(flags, n)
        guard let work = g.zeros(nWork * 4), let cleanFace = g.zeros((n - nWork) * 16), let cleanSrc = g.zeros((n - nWork) * 4) else { return false }
        guard g.run({ enc in
            g.go(enc, "compact_split", n, [.buf(flags), .buf(flags), .buf(triBuf[s]), .buf(srcBuf), .buf(work), .buf(cleanFace),
                                            .buf(cleanSrc), .val(UInt32(n))])
        }) else { return false }
        job.faces[s] = g.read(cleanFace, SIMD3<Int32>.self, n - nWork)
        job.faceSrc[s] = g.read(cleanSrc, Int32.self, n - nWork)
        let workList = g.read(work, UInt32.self, nWork).map { Int($0) }
        let results = SharedBuffer([BooleanJob.CutResult](repeating: BooleanJob.CutResult(), count: workList.count))
        let consS = job.cons[s], splitS = job.splits[s]
        BooleanParallel.chunks(workList.count, minChunk: 16) { r in
            for k in r {
                let t = workList[k]
                let res = BooleanJob.cutOne(tr[t], P, consS[Int32(t)] ?? [], splitS)
                results.withPointer { $0[k] = res }
            }
        }
        for (k, res) in results.items.enumerated() {
            let t = workList[k]
            var ids = res.localId
            if let c = ids.firstIndex(of: -1) { ids[c] = Int32(job.pool.addUnshared(res.centre)) }
            for f in res.tris {
                job.faces[s].append(SIMD3(ids[Int(f.x)], ids[Int(f.y)], ids[Int(f.z)])); job.faceSrc[s].append(job.srcTri[s][t])
            }
            for e in res.constrained { job.cut.insert(CDT2.key(Int(e.x), Int(e.y))) }
            job.stats.constraintFailures += res.failures
        }
        job.stats.cutTriangles += nWork
        return true
    }
}
#endif
