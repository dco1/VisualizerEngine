//  Robust mesh-intersection booleans (ticket item B) — union, difference, intersection, slice.
//
//  The old BSP CSG in `Mesh3+CSG.swift` is unchanged (and keeps its limits). This one:
//   · finds touching triangle pairs with a BVH (or, for big inputs, a GPU all-pairs box test),
//   · intersects them with exact orientation signs (`Robust.orient3d`), coplanar pairs by polygon overlap,
//   · retriangulates every cut triangle with a constrained triangulation that uses exactly the shared curve
//     vertices (`CDT2`), so A's and B's pieces meet vertex for vertex,
//   · classifies each connected patch of pieces (patches are separated by the curve) once, by the generalized
//     winding number of one representative point (GPU for big inputs), or by the coplanar rule below,
//   · emits a welded mesh.
//
//  Guarantees (closed, outward-wound, manifold inputs — `signedVolume > 0`):
//   · the output is index-welded and closed: every edge is shared by exactly two faces (`.byPosition`) — except
//     where the operands genuinely touch along an edge or at a point (non-manifold by nature),
//   · no face is wound inward (the volume is the volume of the boolean), UVs and normals are interpolated across
//     cuts (`.keepCreases` keeps them as they were per face; `.byPosition` shares one vertex per position, first
//     face's UV, normals averaged — hard edges look smooth, use `.keepCreases` to render them),
//   · coplanar faces: where a face of A and a face of B lie in one plane, the overlap is cut out of both and
//     decided by facing — same-facing: union/intersection keep A's, difference drops both; opposite-facing
//     (touching): union/intersection drop both, difference keeps A's. No 0.1 mm nudging needed.
//  Not guaranteed: inputs with self-intersections, open meshes (no inside), operands that touch only along an
//  edge or a vertex (the result then has a non-manifold edge), features thinner than `tolerance`
//  (relative 1e-9 of the joint bounding diagonal by default: points closer than that are one vertex).
//  `BooleanStats.constraintFailures` counts cut curves the retriangulation could not complete (expect 0).

import Foundation
import simd

public enum BooleanOp: Sendable { case union, difference, intersection }

public struct BooleanOptions: Sendable {
    public enum Weld: Sendable { case byPosition, keepCreases }
    public var weld: Weld = .byPosition
    /// Weld tolerance relative to the two meshes' joint bounding-box diagonal.
    public var tolerance: Double = 1e-9
    /// Allow the Metal broadphase / winding kernels (used only for big inputs, and only where a device exists).
    public var useGPU: Bool = true
    /// With `useGPU`: run the GPU-heavy pipeline (grid broadphase, GPU narrow-phase filter, GPU patches and
    /// winding, GPU assembly: `Mesh3+BooleanGPU2*.swift`) on big inputs. False keeps the first GPU path.
    public var useGPUPipeline: Bool = true
    /// Always run the closure audit / safe repairs on the result (otherwise only when the boolean flagged trouble).
    public var audit: Bool = false
    public init() {}
}

public struct BooleanStats: Sendable {
    public var trianglesA = 0, trianglesB = 0
    public var candidatePairs = 0, intersectingPairs = 0, cutTriangles = 0
    public var constraintFailures = 0
    /// Input triangles thinner than the weld tolerance found by the pre-pass, and how many it flipped away
    /// (`Mesh3+BooleanSlivers.swift`).
    public var slivers = 0, sliverFlips = 0
    /// Set when a sliver could not be removed or the post-pass found a defect it could not repair: audit the result.
    public var repairIncomplete = false
    /// Filled by the post-pass (`Mesh3+BooleanAudit.swift`) when it ran: what it removed and what is still wrong.
    public var duplicatesRemoved = 0, openEdges = 0, nonManifoldEdges = 0
    public var patches = 0
    public var broadphaseOnGPU = false, windingOnGPU = false
    public var seconds: [String: Double] = [:]
    public init() {}
}

public struct BooleanResult: Sendable {
    public var mesh: Mesh3
    public var stats: BooleanStats
}

public extension Mesh3 {
    /// A ∪ B. Robust: see the file header.
    func booleanUnion(_ other: Mesh3, options: BooleanOptions = BooleanOptions()) -> Mesh3 {
        boolean(.union, other, options: options).mesh
    }
    /// A − B (carve `other` out of `self`).
    func booleanDifference(_ other: Mesh3, options: BooleanOptions = BooleanOptions()) -> Mesh3 {
        boolean(.difference, other, options: options).mesh
    }
    /// A ∩ B.
    func booleanIntersection(_ other: Mesh3, options: BooleanOptions = BooleanOptions()) -> Mesh3 {
        boolean(.intersection, other, options: options).mesh
    }
    /// Cuts the solid with the plane through `point` with normal `normal`; returns both capped halves
    /// (`above` is on the side the normal points to).
    func slice(point: Vec3, normal: Vec3, options: BooleanOptions = BooleanOptions()) -> (above: Mesh3, below: Mesh3) {
        guard let b = bounds else { return (Mesh3(), Mesh3()) }
        let n = normalize3(normal)
        let helper = abs(n.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let u = normalize3(cross3(helper, n)), v = cross3(n, u)
        let R = 4 * max(len3(b.max - b.min), 1e-6) + abs(dot3(n, point - (b.min + b.max) * 0.5))
        let up = Mesh3.booleanBox(origin: point, u: u, v: v, w: n, half: R, depth: R)
        let down = Mesh3.booleanBox(origin: point, u: u, v: v, w: -n, half: R, depth: R)
        return (booleanIntersection(up, options: options), booleanIntersection(down, options: options))
    }

    /// The general entry point, with statistics.
    func boolean(_ op: BooleanOp, _ other: Mesh3, options: BooleanOptions = BooleanOptions()) -> BooleanResult {
        if options.useGPU, options.useGPUPipeline, BooleanGPU2.shared != nil,
           triangleCount + other.triangleCount >= BooleanGPU2.minTriangles {
            let fast = BooleanJob(self, other, options: options)
            if let mesh = fast.runGPU2(op) { BooleanGPU2.runs += 1; return Self.finished(BooleanResult(mesh: mesh, stats: fast.stats), options) }
            BooleanGPU2.fallbacks += 1
        }
        let job = BooleanJob(self, other, options: options)
        var clock = Date()
        func lap(_ name: String) { job.stats.seconds[name] = Date().timeIntervalSince(clock); clock = Date() }
        job.weld(); lap("weld")
        job.cleanSlivers(); lap("slivers")
        job.cutAll(); lap("cut")
        let rel = [job.classify(0), job.classify(1)]; lap("classify")
        let mesh = job.assemble(op, rel); lap("assemble")
        return Self.finished(BooleanResult(mesh: mesh, stats: job.stats), options)
    }

    private static func finished(_ r: BooleanResult, _ options: BooleanOptions) -> BooleanResult {
        guard options.weld == .byPosition, options.audit || r.stats.repairIncomplete || r.stats.constraintFailures > 0 else { return r }
        var out = r
        out.audit()
        return out
    }

    /// Outward-wound box (12 triangles) spanning `±half` along `u`,`v` and `0…depth` along `w` from `origin`.
    static func booleanBox(origin: Vec3, u: Vec3, v: Vec3, w: Vec3, half: Double, depth: Double) -> Mesh3 {
        var c: [Vec3] = []
        for k in 0..<8 {
            let x = (k & 1) == 0 ? -half : half, y = (k & 2) == 0 ? -half : half, z = (k & 4) == 0 ? 0 : depth
            c.append(origin + u * x + v * y + w * z)
        }
        let centre = origin + w * (depth * 0.5)
        var m = Mesh3()
        let quads = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        for q in quads {
            let mid = (c[q[0]] + c[q[1]] + c[q[2]] + c[q[3]]) * 0.25
            m.addQuad(c[q[0]], c[q[1]], c[q[2]], c[q[3]], outward: mid - centre)
        }
        return m
    }
}

// MARK: classification

enum BooleanRel { case inside, outside, same, opposite }

extension BooleanJob {
    func faceNormal(_ f: SIMD3<Int32>, _ P: [Vec3]) -> Vec3 {
        cross3(P[Int(f.y)] - P[Int(f.x)], P[Int(f.z)] - P[Int(f.x)])
    }

    /// Relation of every face of side `s` to the other mesh's solid.
    func classify(_ s: Int) -> [BooleanRel] {
        let o = 1 - s
        let P = pool.points
        let fs = faces[s]
        var parent = (0..<fs.count).map { Int32($0) }
        func find(_ x: Int32) -> Int32 {
            var r = x
            while parent[Int(r)] != r { r = parent[Int(r)] }
            var c = x
            while parent[Int(c)] != r { let n = parent[Int(c)]; parent[Int(c)] = r; c = n }
            return r
        }
        var edgeFace: [UInt64: Int32] = [:]
        edgeFace.reserveCapacity(fs.count * 2)
        for (i, f) in fs.enumerated() {
            for k in 0..<3 {
                let a = Int(f[k]), b = Int(f[(k + 1) % 3])
                let key = CDT2.key(a, b)
                if cut.contains(key) { continue }
                if let g = edgeFace[key] {
                    let ra = find(Int32(i)), rb = find(g)
                    if ra != rb { parent[Int(ra)] = rb }
                } else { edgeFace[key] = Int32(i) }
            }
        }
        var best: [Int32: (Int32, Double)] = [:]
        for i in 0..<fs.count {
            let r = find(Int32(i))
            let area = len3(faceNormal(fs[i], P))
            if let b = best[r], b.1 >= area { continue }
            best[r] = (Int32(i), area)
        }
        stats.patches += best.count
        let roots = Array(best.keys)

        let (lo, hi) = boxes(o, P)
        let bvh = BoxBVH(lo: lo, hi: hi)
        let to = tris[o]
        var relOf: [Int32: BooleanRel] = [:]
        var windPts: [Vec3] = [], windRoots: [Int32] = []
        for r in roots {
            let f = fs[Int(best[r]!.0)]
            let c = (P[Int(f.x)] + P[Int(f.y)] + P[Int(f.z)]) / 3
            let nF = faceNormal(f, P)
            var cop: BooleanRel?
            bvh.query(c - Vec3(repeating: eps), c + Vec3(repeating: eps)) { t in
                if cop != nil { return }
                let v = to[t]
                let a = P[Int(v.x)], b = P[Int(v.y)], cc = P[Int(v.z)]
                let nS = cross3(b - a, cc - a)
                let nu = normalize3(nS)
                if abs(dot3(nu, c - a)) > self.eps { return }
                let pts = [a, b, cc]
                for e in 0..<3 {
                    let m = normalize3(cross3(nu, pts[(e + 1) % 3] - pts[e]))
                    if dot3(m, c - pts[e]) < -self.eps { return }
                }
                cop = dot3(nS, nF) > 0 ? .same : .opposite
            }
            if let c = cop { relOf[r] = c } else { windPts.append(c); windRoots.append(r) }
        }
        let w = windingNumbers(windPts, side: o, P)
        for (i, r) in windRoots.enumerated() { relOf[r] = abs(w[i]) > 0.5 ? .inside : .outside }
        return (0..<fs.count).map { relOf[find(Int32($0))]! }
    }

    private func windingNumbers(_ pts: [Vec3], side o: Int, _ P: [Vec3]) -> [Double] {
        guard !pts.isEmpty else { return [] }
        var corners: [Vec3] = []
        corners.reserveCapacity(tris[o].count * 3)
        for v in tris[o] { corners.append(P[Int(v.x)]); corners.append(P[Int(v.y)]); corners.append(P[Int(v.z)]) }
        if options.useGPU, pts.count * tris[o].count >= 200_000, let g = BooleanGPU.shared,
           let w = g.winding(points: pts, corners: corners) {
            stats.windingOnGPU = true
            return w.map { Double($0) }
        }
        return BooleanJob.windingCPU(pts, corners)
    }

    static func windingCPU(_ pts: [Vec3], _ corners: [Vec3]) -> [Double] {
        let out = SharedBuffer([Double](repeating: 0, count: pts.count))
        BooleanParallel.chunks(pts.count, minChunk: 1) { r in
            for i in r {
                let p = pts[i]
                var sum = 0.0
                var t = 0
                while t < corners.count {
                    let a = corners[t] - p, b = corners[t + 1] - p, c = corners[t + 2] - p
                    let la = len3(a), lb = len3(b), lc = len3(c)
                    let num = dot3(a, cross3(b, c))
                    let den = la * lb * lc + dot3(a, b) * lc + dot3(b, c) * la + dot3(c, a) * lb
                    sum += 2 * atan2(num, den)
                    t += 3
                }
                out.withPointer { $0[i] = sum / (4 * Double.pi) }
            }
        }
        return out.items
    }

    // MARK: assembly

    private func keep(_ op: BooleanOp, side: Int, _ r: BooleanRel) -> (keep: Bool, flip: Bool) {
        switch (op, side) {
        case (.union, _):          return (r == .outside || (side == 0 && r == .same), false)
        case (.intersection, _):   return (r == .inside || (side == 0 && r == .same), false)
        case (.difference, 0):     return (r == .outside || r == .opposite, false)
        case (.difference, _):     return (r == .inside, true)
        }
    }

    func assemble(_ op: BooleanOp, _ rel: [[BooleanRel]]) -> Mesh3 {
        let P = pool.points
        var out = Mesh3()
        var normalSum: [Vec3] = []
        var byPos: [Int32: Int] = [:]
        struct Key: Hashable { var id: Int32; var n: SIMD3<Int32>; var uv: SIMD2<Int32> }
        var byKey: [Key: Int] = [:]
        for s in 0..<2 {
            let m = meshes[s]
            let hasN = m.normals.count == m.positions.count, hasUV = m.uvs.count == m.positions.count
            for (i, f) in faces[s].enumerated() {
                let (k, flip) = keep(op, side: s, rel[s][i])
                guard k else { continue }
                let src = Int(faceSrc[s][i])
                let i0 = Int(m.indices[src * 3]), i1 = Int(m.indices[src * 3 + 1]), i2 = Int(m.indices[src * 3 + 2])
                let a = m.positions[i0], b = m.positions[i1], c = m.positions[i2]
                let fn = normalize3(cross3(b - a, c - a))
                var idx: [UInt32] = []
                for corner in [f.x, f.y, f.z] {
                    let p = P[Int(corner)]
                    let v0 = b - a, v1 = c - a, v2 = p - a
                    let d00 = dot3(v0, v0), d01 = dot3(v0, v1), d11 = dot3(v1, v1)
                    let d20 = dot3(v2, v0), d21 = dot3(v2, v1)
                    let den = d00 * d11 - d01 * d01
                    let bv = den != 0 ? (d11 * d20 - d01 * d21) / den : 0
                    let bw = den != 0 ? (d00 * d21 - d01 * d20) / den : 0
                    let bu = 1 - bv - bw
                    var n = hasN ? normalize3(m.normals[i0] * bu + m.normals[i1] * bv + m.normals[i2] * bw) : fn
                    if len3(n) == 0 { n = fn }
                    if flip { n = -n }
                    let uv = hasUV ? m.uvs[i0] * bu + m.uvs[i1] * bv + m.uvs[i2] * bw : Vec2.zero
                    switch options.weld {
                    case .byPosition:
                        if let j = byPos[corner] { normalSum[j] += n; idx.append(UInt32(j)) } else {
                            byPos[corner] = out.positions.count
                            idx.append(UInt32(out.positions.count))
                            out.positions.append(p); out.uvs.append(uv); normalSum.append(n)
                        }
                    case .keepCreases:
                        let key = Key(id: corner,
                                      n: SIMD3(Int32((n.x * 1e4).rounded()), Int32((n.y * 1e4).rounded()), Int32((n.z * 1e4).rounded())),
                                      uv: SIMD2(Int32((uv.x * 1e6).rounded()), Int32((uv.y * 1e6).rounded())))
                        if let j = byKey[key] { idx.append(UInt32(j)) } else {
                            byKey[key] = out.positions.count
                            idx.append(UInt32(out.positions.count))
                            out.positions.append(p); out.uvs.append(uv); out.normals.append(n)
                        }
                    }
                }
                if idx[0] == idx[1] || idx[1] == idx[2] || idx[0] == idx[2] { continue }
                if flip { idx.swapAt(1, 2) }
                out.indices.append(contentsOf: idx)
            }
        }
        if options.weld == .byPosition { out.normals = normalSum.map { normalize3($0) } }
        return out
    }
}
