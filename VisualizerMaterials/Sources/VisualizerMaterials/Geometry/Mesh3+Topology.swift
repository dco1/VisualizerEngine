//  TriTopology — the mutable indexed triangle mesh that decimation, remeshing and hole filling edit.
//
//  Guarantees: vertices welded by position, per-corner UVs kept (so UV seams are detectable), vertex → face
//  adjacency kept in step with every edit, original winding preserved, crease edges (dihedral above a threshold)
//  remembered. `toMesh3` rebuilds a render mesh: vertices split per (position, smooth group, UV), normals
//  angle-weighted per group, so hard edges and UV seams come back as split vertices.
//  Does not guarantee: anything for non-manifold input beyond "edges with ≠ 2 faces are treated as boundary".
//
//  CPU: this is graph surgery (pointer chasing, priority queues), not per-vertex number crunching, so it stays
//  off the GPU; the per-vertex projection step of the remesher is the part a GPU port would take.

import simd

struct TriTopology {
    var pos: [Vec3]
    var tri: [SIMD3<Int>] = []
    var cuv: [Vec2] = []              // three per triangle, in corner order
    var alive: [Bool] = []
    var vf: [[Int]]                   // vertex → incident live triangles
    var creases: Set<Int64> = []      // sharp edges (by `key`), remembered through edits
    var aliveCount = 0

    static func key(_ a: Int, _ b: Int) -> Int64 { Int64(Swift.min(a, b)) << 32 | Int64(Swift.max(a, b)) }

    /// Weld `m` by position (`weldDistance`, default relative to its size) and mark creases sharper than `creaseDegrees`.
    init(_ m: Mesh3, weldDistance: Double? = nil, creaseDegrees: Double = 40) {
        var diag = 1.0
        if let b = m.bounds { diag = Swift.max(len3(b.max - b.min), 1e-9) }
        let rep = Mesh3.weldClusters(m.positions, distance: weldDistance ?? diag * 1e-7)
        var remap = [Int](repeating: -1, count: m.positions.count)
        pos = []
        for i in 0 ..< m.positions.count where rep[i] == i { remap[i] = pos.count; pos.append(m.positions[i]) }
        for i in 0 ..< m.positions.count { remap[i] = remap[rep[i]] }
        vf = Array(repeating: [], count: pos.count)
        let hasUV = m.uvs.count == m.positions.count
        var t = 0
        while t + 2 < m.indices.count {
            let i0 = Int(m.indices[t]), i1 = Int(m.indices[t + 1]), i2 = Int(m.indices[t + 2])
            let a = remap[i0], b = remap[i1], c = remap[i2]
            if a != b, b != c, a != c, len3(cross3(pos[b] - pos[a], pos[c] - pos[a])) > 1e-18 {
                addFace(a, b, c, hasUV ? m.uvs[i0] : .zero, hasUV ? m.uvs[i1] : .zero, hasUV ? m.uvs[i2] : .zero)
            }
            t += 3
        }
        markCreases(degrees: creaseDegrees)
    }

    @discardableResult
    mutating func addFace(_ a: Int, _ b: Int, _ c: Int, _ ua: Vec2 = .zero, _ ub: Vec2 = .zero, _ uc: Vec2 = .zero) -> Int {
        let f = tri.count
        tri.append(SIMD3(a, b, c)); cuv.append(ua); cuv.append(ub); cuv.append(uc); alive.append(true)
        vf[a].append(f); vf[b].append(f); vf[c].append(f); aliveCount += 1
        return f
    }

    mutating func addVertex(_ p: Vec3) -> Int { pos.append(p); vf.append([]); return pos.count - 1 }

    mutating func kill(_ f: Int) {
        guard alive[f] else { return }
        alive[f] = false; aliveCount -= 1
        for k in 0 ..< 3 { let v = tri[f][k]; if let i = vf[v].firstIndex(of: f) { vf[v].remove(at: i) } }
    }

    // MARK: queries

    func corner(_ f: Int, _ v: Int) -> Int { tri[f].x == v ? 0 : (tri[f].y == v ? 1 : 2) }
    func uv(_ f: Int, _ v: Int) -> Vec2 { cuv[f * 3 + corner(f, v)] }
    func contains(_ f: Int, _ v: Int) -> Bool { tri[f].x == v || tri[f].y == v || tri[f].z == v }
    func facesOn(_ a: Int, _ b: Int) -> [Int] { vf[a].filter { contains($0, b) } }
    func third(_ f: Int, _ a: Int, _ b: Int) -> Int {
        let t = tri[f]; return (t.x != a && t.x != b) ? t.x : ((t.y != a && t.y != b) ? t.y : t.z)
    }
    func neighbors(_ v: Int) -> [Int] {
        var out: [Int] = []
        for f in vf[v] { for k in 0 ..< 3 { let w = tri[f][k]; if w != v, !out.contains(w) { out.append(w) } } }
        return out
    }
    /// Unnormalised normal of triangle `f` (area × 2).
    func rawNormal(_ f: Int) -> Vec3 {
        let t = tri[f]; return cross3(pos[t.y] - pos[t.x], pos[t.z] - pos[t.x])
    }
    func edgeLength(_ a: Int, _ b: Int) -> Double { len3(pos[a] - pos[b]) }

    /// Sharp (≥ `degrees`) interior edges become creases. Boundary edges are found on the fly instead.
    mutating func markCreases(degrees: Double) {
        let cosC = cos(degrees * .pi / 180)
        for f in 0 ..< tri.count where alive[f] {
            for k in 0 ..< 3 {
                let a = tri[f][k], b = tri[f][(k + 1) % 3]
                guard a < b else { continue }
                let sh = facesOn(a, b)
                if sh.count == 2, dot3(normalize3(rawNormal(sh[0])), normalize3(rawNormal(sh[1]))) < cosC {
                    creases.insert(TriTopology.key(a, b))
                }
            }
        }
    }

    // MARK: output

    /// A render mesh: vertices split per (position, smooth group, UV). `creaseDegrees` decides the smooth groups.
    func toMesh3(creaseDegrees: Double = 40) -> Mesh3 {
        let cosC = cos(creaseDegrees * .pi / 180)
        var fn = [Vec3](repeating: .zero, count: tri.count)
        for f in 0 ..< tri.count where alive[f] { fn[f] = normalize3(rawNormal(f)) }
        struct Key: Hashable { var v: Int, g: Int, u: Int64, w: Int64 }
        var ids: [Key: UInt32] = [:]
        var out = Mesh3()
        var cornerIndex = [UInt32](repeating: 0, count: tri.count * 3)
        for v in 0 ..< pos.count {
            let fs = vf[v]
            guard !fs.isEmpty else { continue }
            var parent = Array(0 ..< fs.count)
            func root(_ i: Int) -> Int { var r = i; while parent[r] != r { r = parent[r] }; return r }
            for i in 0 ..< fs.count { for j in (i + 1) ..< fs.count {
                let ti = tri[fs[i]], tj = tri[fs[j]]
                var common = 0
                for k in 0 ..< 3 { let w = ti[k]; if w != v, tj.x == w || tj.y == w || tj.z == w { common += 1 } }
                if common == 1, dot3(fn[fs[i]], fn[fs[j]]) >= cosC { parent[root(i)] = root(j) }
            } }
            var groupNormal: [Int: Vec3] = [:]
            for (i, f) in fs.enumerated() {
                let t = tri[f], k = corner(f, v)
                let a = pos[t[(k + 1) % 3]] - pos[v], b = pos[t[(k + 2) % 3]] - pos[v]
                let la = len3(a), lb = len3(b)
                let ang = (la > 1e-15 && lb > 1e-15) ? acos(Swift.max(-1, Swift.min(1, dot3(a, b) / (la * lb)))) : 0
                groupNormal[root(i), default: .zero] += fn[f] * ang
            }
            for (i, f) in fs.enumerated() {
                let c = cuv[f * 3 + corner(f, v)], g = root(i)
                let key = Key(v: v, g: g, u: Int64((c.x * 1e6).rounded()), w: Int64((c.y * 1e6).rounded()))
                if ids[key] == nil {
                    ids[key] = UInt32(out.positions.count)
                    out.positions.append(pos[v]); out.uvs.append(c)
                    let n = groupNormal[g]!
                    out.normals.append(len3(n) > 1e-12 ? normalize3(n) : fn[f])
                }
                cornerIndex[f * 3 + corner(f, v)] = ids[key]!
            }
        }
        for f in 0 ..< tri.count where alive[f] {
            for k in 0 ..< 3 { out.indices.append(cornerIndex[f * 3 + k]) }
        }
        return out
    }
}
