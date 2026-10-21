//  Loop subdivision with creases — a coarse CAGE becomes a smooth surface, with hard edges kept where you
//  want them.
//
//  This is the modeller's "block it out rough, then smooth it" workflow (Blender's Subdivision Surface
//  modifier with crease weights). Hand-sampling every curve of a soap bar, a rolled arm, a faucet body or a
//  bull-nosed counter to the point it looks round costs hundreds of stations; a 40-triangle cage and two
//  levels of this costs nothing.
//
//   · The mesh is welded by position first, so the topology is the surface's, not the triangle soup's.
//   · An edge is a CREASE when it is sharper than `creaseDegrees`, or lies on a boundary or non-manifold join.
//     Crease edges subdivide to their midpoint and their vertices slide only along the crease, so a box
//     keeps its twelve edges while its faces round; vertices where ≠ 2 creases meet (a box corner) stay put.
//   · Smooth interior edge point = 3/8 (a + b) + 1/8 (c + d); smooth vertex = (1 − nβ) v + β Σ neighbours,
//     β = 3/16 (n = 3) else 3/(8n) — the standard Loop weights, so the limit is C² away from creases.
//   · Normals are rebuilt (angle-weighted, creased at the same threshold). The contract of the result is
//     `MeshAudit.isSoundSmooth`.
//
//  ⚠️ UVs are NOT carried: welding merges seams, and the new vertices have no authored coordinates, so the result
//  has zero UVs. Use it for geometry that takes a procedural / vertex-coloured material, or re-project UVs after.
//  Each level quadruples the triangle count — 2 levels is almost always enough; 4 is the cap.
//  Shrinkage: a closed smooth surface contracts toward its interior (a cube becomes a sphere inside the cube) —
//  size the cage a little LARGE, or add creases where the original size must be kept.

import simd

public extension Mesh3 {

    /// Subdivide `levels` times (1…4). Edges sharper than `creaseDegrees` stay sharp.
    func subdivided(levels: Int = 2, creaseDegrees: Double = 35) -> Mesh3 {
        var s = LoopSubdivider(self, creaseDegrees: creaseDegrees)
        guard s.valid else { return self }
        for _ in 0..<Swift.max(0, Swift.min(4, levels)) { s.step() }
        return s.build().smoothed(creaseDegrees: Swift.max(creaseDegrees, 20))
    }
}

private struct LoopSubdivider {
    struct Edge: Hashable { let a: Int, b: Int
        init(_ x: Int, _ y: Int) { a = Swift.min(x, y); b = Swift.max(x, y) } }

    var pos: [Vec3] = []
    var faces: [[Int]] = []                 // triangles, original winding
    var creases: Set<Edge> = []
    var valid = false

    init(_ m: Mesh3, creaseDegrees: Double) {
        var map: [SIMD3<Int64>: Int] = [:]
        var ids: [Int] = []
        for p in m.positions {
            let k = SIMD3<Int64>(Int64((p.x * 1e6).rounded()), Int64((p.y * 1e6).rounded()), Int64((p.z * 1e6).rounded()))
            if let id = map[k] { ids.append(id) } else { map[k] = pos.count; ids.append(pos.count); pos.append(p) }
        }
        var t = 0
        while t + 2 < m.indices.count {
            let f = [ids[Int(m.indices[t])], ids[Int(m.indices[t + 1])], ids[Int(m.indices[t + 2])]]
            if Set(f).count == 3 { faces.append(f) }
            t += 3
        }
        valid = !faces.isEmpty
        guard valid else { return }
        var byEdge: [Edge: [Int]] = [:]
        for (i, f) in faces.enumerated() {
            for k in 0..<3 { byEdge[Edge(f[k], f[(k + 1) % 3]), default: []].append(i) }
        }
        func normal(_ f: [Int]) -> Vec3 { normalize3(cross3(pos[f[1]] - pos[f[0]], pos[f[2]] - pos[f[0]])) }
        let cosC = cos(creaseDegrees * .pi / 180)
        for (e, fs) in byEdge {
            if fs.count != 2 || dot3(normal(faces[fs[0]]), normal(faces[fs[1]])) < cosC { creases.insert(e) }
        }
    }

    mutating func step() {
        // topology
        var edgeFaces: [Edge: [Int]] = [:]
        var neighbours: [Int: Set<Int>] = [:]
        for (i, f) in faces.enumerated() {
            for k in 0..<3 {
                let a = f[k], b = f[(k + 1) % 3]
                edgeFaces[Edge(a, b), default: []].append(i)
                neighbours[a, default: []].insert(b); neighbours[b, default: []].insert(a)
            }
        }
        // edge points
        var newPos = pos
        var edgePoint: [Edge: Int] = [:]
        for (e, fs) in edgeFaces {
            var p = (pos[e.a] + pos[e.b]) * 0.5
            if !creases.contains(e), fs.count == 2 {
                func opposite(_ f: [Int]) -> Vec3 { pos[f.first { $0 != e.a && $0 != e.b }!] }
                let along: Vec3 = (pos[e.a] + pos[e.b]) * 0.375
                let across: Vec3 = (opposite(faces[fs[0]]) + opposite(faces[fs[1]])) * 0.125
                p = along + across
            }
            edgePoint[e] = newPos.count
            newPos.append(p)
        }
        // vertex points
        var creaseNeighbours: [Int: [Int]] = [:]
        for e in creases { creaseNeighbours[e.a, default: []].append(e.b); creaseNeighbours[e.b, default: []].append(e.a) }
        for v in 0..<pos.count {
            guard let nb = neighbours[v], !nb.isEmpty else { continue }
            if let cn = creaseNeighbours[v] {
                if cn.count == 2 {
                    let keep: Vec3 = pos[v] * 0.75
                    let slide: Vec3 = (pos[cn[0]] + pos[cn[1]]) * 0.125
                    newPos[v] = keep + slide
                }
                // else: a corner (1 or ≥ 3 creases) — it stays where the cage put it
            } else {
                let n = Double(nb.count)
                let beta = nb.count == 3 ? 3.0 / 16.0 : 3.0 / (8.0 * n)
                let sum: Vec3 = nb.reduce(Vec3.zero) { $0 + pos[$1] }
                let keep: Vec3 = pos[v] * (1 - n * beta)
                newPos[v] = keep + sum * beta
            }
        }
        // faces
        var newFaces: [[Int]] = []
        newFaces.reserveCapacity(faces.count * 4)
        for f in faces {
            let ab = edgePoint[Edge(f[0], f[1])]!, bc = edgePoint[Edge(f[1], f[2])]!, ca = edgePoint[Edge(f[2], f[0])]!
            newFaces.append([f[0], ab, ca]); newFaces.append([f[1], bc, ab])
            newFaces.append([f[2], ca, bc]); newFaces.append([ab, bc, ca])
        }
        // a crease edge splits into two crease edges
        var newCreases: Set<Edge> = []
        for e in creases {
            if let m = edgePoint[e] { newCreases.insert(Edge(e.a, m)); newCreases.insert(Edge(m, e.b)) }
        }
        pos = newPos; faces = newFaces; creases = newCreases
    }

    func build() -> Mesh3 {
        var m = Mesh3()
        for f in faces {
            let n = cross3(pos[f[1]] - pos[f[0]], pos[f[2]] - pos[f[0]])
            guard len3(n) > 1e-14 else { continue }
            m.addTriangle(pos[f[0]], pos[f[1]], pos[f[2]], outward: n)       // winding is the cage's own
        }
        return m
    }
}
