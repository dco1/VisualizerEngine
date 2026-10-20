//  DH-1437 — general edge bevel / fillet for any Mesh3.
//
//  `addBeveledBox` rounds ONE shape. This rounds the sharp edges of ANY triangle mesh: a fork,
//  a knife, a hinge, a faucet body. A real edge is never razor-sharp — a ~0.3 mm ease is what
//  catches a highlight and makes metal read as metal — and until now the only softening on offer
//  was smooth normals, which fake the shading but leave the SILHOUETTE sharp.
//
//  Method (all deterministic, no search):
//   1. Weld by position. An edge is SHARP when exactly two faces meet at more than
//      `creaseDegrees`. Sharp edges that dead-end at an interior vertex are dropped (they would
//      leave a crack); a bevel must run edge-to-edge or close on a corner.
//   2. Faces joined across non-sharp edges form a PATCH (a flat panel, or a smooth curved skin).
//      Every vertex of a patch slides INWARD, inside the patch, until it is `radius` clear of each
//      sharp edge that touches it. One edge → slide perpendicular to it; two or more → the
//      least-squares point clear of all of them (an exact corner for two).
//   3. Each sharp edge is bridged by a strip between its two patches' moved edges, bowed through
//      the original edge as a quadratic Bézier — a round, not a chamfer — with normals swung from
//      one patch's to the other's, so even `segments: 1` shades as a curve.
//   4. A vertex where three or more bevels meet is capped with a fan over the ring their
//      cross-sections make, pulled back toward the original corner so a box corner is a dome.
//
//  Every triangle goes through `addTriangleSmooth`, so winding stays declared once. The contract of
//  the result is `MeshAudit.isSoundSmooth` (smooth normals), like `Mesh3.smoothed`.
//
//  ⚠️ `radius` must be under HALF the thinnest section it bevels (a 2.4 mm tine takes ≤ 1 mm):
//  two opposite bevels on one face would cross. Per-vertex clamping to 40 % of the shortest
//  incident edge stops the worst of it, but the thickness is the caller's to know.

import simd

public extension Mesh3 {

    /// Round every edge sharper than `creaseDegrees` by `radius` (metres).
    func beveled(radius: Double, creaseDegrees: Double = 30, segments: Int = 3) -> Mesh3 {
        var bevel = MeshBeveler(mesh: self, radius: radius, creaseDegrees: creaseDegrees,
                                segments: Swift.max(1, segments))
        return bevel.run()
    }
}

private struct MeshBeveler {
    let src: Mesh3
    let radius: Double
    let cosCrease: Double
    let segments: Int

    struct Edge: Hashable { let a: Int, b: Int
        init(_ x: Int, _ y: Int) { a = Swift.min(x, y); b = Swift.max(x, y) } }
    struct PV: Hashable { let patch: Int, w: Int }
    struct Corner { var pos: Vec3; var normal: Vec3; var uv: Vec2 }

    var wid: [Int] = []                   // original vertex → welded id
    var wpos: [Vec3] = []                 // welded id → position
    var faces: [(w: [Int], c: [Int])] = []   // welded ids + original corner indices per face
    var fn: [Vec3] = []                   // geometric face normals
    var patchOf: [Int] = []
    var sharp: [Edge: [Int]] = [:]        // sharp edge → its two faces
    var moved: [PV: Corner] = [:]

    init(mesh: Mesh3, radius: Double, creaseDegrees: Double, segments: Int) {
        src = mesh; self.radius = radius; self.segments = segments
        cosCrease = cos(creaseDegrees * .pi / 180)
    }

    mutating func run() -> Mesh3 {
        guard radius > 1e-9, src.indices.count >= 3 else { return src }
        weld()
        findSharpEdges()
        guard !sharp.isEmpty else { return src }
        buildPatches()
        placeCorners()
        var out = Mesh3()
        emitPatches(into: &out)
        emitStrips(into: &out)
        emitCaps(into: &out)
        return out
    }

    // MARK: weld + sharp edges

    mutating func weld() {
        var map: [SIMD3<Int64>: Int] = [:]
        let q = 1.0e6
        wid = src.positions.map { p in
            let k = SIMD3<Int64>(Int64((p.x * q).rounded()), Int64((p.y * q).rounded()),
                                 Int64((p.z * q).rounded()))
            if let id = map[k] { return id }
            let id = wpos.count; wpos.append(p); map[k] = id; return id
        }
        var t = 0
        while t + 2 < src.indices.count {
            let c = [Int(src.indices[t]), Int(src.indices[t + 1]), Int(src.indices[t + 2])]
            let w = c.map { wid[$0] }
            let n = cross3(wpos[w[1]] - wpos[w[0]], wpos[w[2]] - wpos[w[0]])
            if len3(n) > 1e-14, Set(w).count == 3 {
                faces.append((w, c)); fn.append(normalize3(n))
            }
            t += 3
        }
    }

    mutating func findSharpEdges() {
        var byEdge: [Edge: [Int]] = [:]
        for (f, face) in faces.enumerated() {
            for i in 0..<3 { byEdge[Edge(face.w[i], face.w[(i + 1) % 3]), default: []].append(f) }
        }
        sharpCandidates = byEdge
        for (e, fs) in byEdge where fs.count == 2 && dot3(fn[fs[0]], fn[fs[1]]) < cosCrease {
            sharp[e] = fs
        }
        // Drop dead-end bevels at interior vertices — they would leave a crack.
        var open = Set<Int>()                              // vertices on a boundary / non-manifold edge
        for (e, fs) in byEdge where fs.count != 2 { open.insert(e.a); open.insert(e.b) }
        var changed = true
        while changed {
            changed = false
            var deg: [Int: Int] = [:]
            for e in sharp.keys { deg[e.a, default: 0] += 1; deg[e.b, default: 0] += 1 }
            for e in Array(sharp.keys) where
                (deg[e.a] == 1 && !open.contains(e.a)) || (deg[e.b] == 1 && !open.contains(e.b)) {
                sharp[e] = nil; changed = true
            }
        }
    }
    var sharpCandidates: [Edge: [Int]] = [:]

    // MARK: patches

    mutating func buildPatches() {
        var parent = Array(0..<faces.count)
        func find(_ x: Int) -> Int { var r = x; while parent[r] != r { parent[r] = parent[parent[r]]; r = parent[r] }; return r }
        for (e, fs) in sharpCandidates where sharp[e] == nil {
            for i in 1..<fs.count { parent[find(fs[i])] = find(fs[0]) }
        }
        var ids: [Int: Int] = [:]
        patchOf = faces.indices.map { f in
            let r = find(f); if let id = ids[r] { return id }
            let id = ids.count; ids[r] = id; return id
        }
    }

    // MARK: moved corners

    func clampedRadius(at w: Int, face f: Int) -> Double {
        var shortest = Double.infinity
        for i in 0..<3 {
            let a = faces[f].w[i], b = faces[f].w[(i + 1) % 3]
            if a == w || b == w { shortest = Swift.min(shortest, len3(wpos[a] - wpos[b])) }
        }
        return Swift.min(radius, 0.4 * shortest)
    }

    mutating func placeCorners() {
        // constraint directions per (patch, vertex)
        var cons: [PV: [(d: Vec3, r: Double)]] = [:]
        for (e, fs) in sharp {
            for f in fs {
                let p = patchOf[f]
                let third = faces[f].w.first { $0 != e.a && $0 != e.b }!
                for (v, u) in [(e.a, e.b), (e.b, e.a)] {
                    let eh = normalize3(wpos[u] - wpos[v])
                    let t = wpos[third] - wpos[v]
                    let d = normalize3(t - eh * dot3(t, eh))
                    cons[PV(patch: p, w: v), default: []].append((d, clampedRadius(at: v, face: f)))
                }
            }
        }
        for (f, face) in faces.enumerated() {
            let p = patchOf[f]
            for i in 0..<3 {
                let key = PV(patch: p, w: face.w[i])
                guard moved[key] == nil else { continue }
                var pos = wpos[face.w[i]]
                if let list = cons[key] { pos += solve(list) }
                let c = face.c[i]
                moved[key] = Corner(pos: pos, normal: src.normals.count > c ? src.normals[c] : fn[f],
                                    uv: src.uvs.count > c ? src.uvs[c] : .zero)
            }
        }
    }

    /// Least-squares slide `δ` with `dot(δ, dᵢ) = rᵢ`; exact for one or two independent edges.
    func solve(_ list: [(d: Vec3, r: Double)]) -> Vec3 {
        if list.count == 1 { return list[0].d * list[0].r }
        var g = simd_double3x3(0), rhs = Vec3.zero
        for (d, r) in list {
            g += simd_double3x3(columns: (d * d.x, d * d.y, d * d.z))
            rhs += d * r
        }
        g += simd_double3x3(diagonal: Vec3(repeating: 1e-6))
        return g.inverse * rhs
    }

    // MARK: emit

    func emitPatches(into out: inout Mesh3) {
        for (f, face) in faces.enumerated() {
            let p = patchOf[f]
            let cs = (0..<3).map { moved[PV(patch: p, w: face.w[$0])]! }
            out.addTriangleSmooth(cs[0].pos, cs[1].pos, cs[2].pos, outward: fn[f],
                                  normals: (cs[0].normal, cs[1].normal, cs[2].normal),
                                  uv: (cs[0].uv, cs[1].uv, cs[2].uv))
        }
    }

    /// The cross-section from patch `pa`'s side of vertex `w` to patch `pb`'s, `segments + 1` points.
    func arc(_ w: Int, _ pa: Int, _ pb: Int) -> [Corner] {
        let a = moved[PV(patch: pa, w: w)]!, b = moved[PV(patch: pb, w: w)]!
        let v = wpos[w]
        return (0...segments).map { i in
            let t = Double(i) / Double(segments), s = 1 - t
            return Corner(pos: a.pos * (s * s) + v * (2 * s * t) + b.pos * (t * t),
                          normal: normalize3(a.normal * s + b.normal * t),
                          uv: a.uv * s + b.uv * t)
        }
    }

    func emitStrips(into out: inout Mesh3) {
        for (e, fs) in sharp {
            let pa = patchOf[fs[0]], pb = patchOf[fs[1]]
            let arcA = arc(e.a, pa, pb), arcB = arc(e.b, pa, pb)
            let outward = normalize3(fn[fs[0]] + fn[fs[1]])
            for i in 0..<segments {
                let q = [arcA[i], arcA[i + 1], arcB[i + 1], arcB[i]]
                out.addTriangleSmooth(q[0].pos, q[1].pos, q[2].pos, outward: outward,
                                      normals: (q[0].normal, q[1].normal, q[2].normal),
                                      uv: (q[0].uv, q[1].uv, q[2].uv))
                out.addTriangleSmooth(q[0].pos, q[2].pos, q[3].pos, outward: outward,
                                      normals: (q[0].normal, q[2].normal, q[3].normal),
                                      uv: (q[0].uv, q[2].uv, q[3].uv))
            }
        }
    }

    func emitCaps(into out: inout Mesh3) {
        var at: [Int: [Edge]] = [:]
        for e in sharp.keys { at[e.a, default: []].append(e); at[e.b, default: []].append(e) }
        for (w, edges) in at {
            let patchPairs = edges.map { e -> Set<Int> in Set(sharp[e]!.map { patchOf[$0] }) }
            if edges.count == 2, patchPairs[0] == patchPairs[1] { continue }   // a straight run
            guard edges.count >= 2 else { continue }
            let v = wpos[w]
            var axis = Vec3.zero
            for e in edges { for f in sharp[e]! { axis += fn[f] } }
            axis = normalize3(axis)
            // order the bevels around the corner by their direction, seen along the corner's axis
            let ref = abs(axis.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
            let u = normalize3(cross3(axis, ref)), vv = cross3(axis, u)
            func angle(_ e: Edge) -> Double {
                let o = wpos[e.a == w ? e.b : e.a] - v
                return atan2(dot3(o, vv), dot3(o, u))
            }
            let ordered = edges.sorted { angle($0) < angle($1) }
            var ring: [Corner] = []
            for i in 0..<ordered.count {
                let prev = Set(sharp[ordered[(i + ordered.count - 1) % ordered.count]]!.map { patchOf[$0] })
                let next = Set(sharp[ordered[(i + 1) % ordered.count]]!.map { patchOf[$0] })
                let mine = Set(sharp[ordered[i]]!.map { patchOf[$0] })
                guard let from = mine.intersection(prev).first, let to = mine.intersection(next).first,
                      from != to || mine.count == 1 else { continue }
                let pts = arc(w, from, to)
                ring.append(contentsOf: i == 0 ? pts : Array(pts.dropFirst()))
            }
            guard ring.count >= 3 else { continue }
            let mean = ring.reduce(Vec3.zero) { $0 + $1.pos } / Double(ring.count)
            let centre = Corner(pos: mix(mean, v, 0.6),
                                normal: axis, uv: ring.reduce(Vec2.zero) { $0 + $1.uv } / Double(ring.count))
            for i in 0..<ring.count {
                let a = ring[i], b = ring[(i + 1) % ring.count]
                guard len3(a.pos - b.pos) > 1e-9 else { continue }
                out.addTriangleSmooth(centre.pos, a.pos, b.pos, outward: axis,
                                      normals: (centre.normal, a.normal, b.normal),
                                      uv: (centre.uv, a.uv, b.uv))
            }
        }
    }
}
