//  Solidify — give a surface thickness.
//
//  Two layers: A (the original winding, facing out) and B (reversed, facing in), each moved along the welded smooth
//  vertex normal; every boundary edge is closed with a rim quad. Offset -1…1 slides the shell: -1 keeps the original
//  as the outer skin and grows inward, +1 grows outward, 0 straddles it:
//      A = p + n·t·(1+offset)/2     B = p + n·t·(offset-1)/2
//  Even thickness stretches each vertex's offset by 1/mean(cos) of the faces round it, so a wall turning a corner
//  keeps its thickness rather than thinning to t·cos(half-angle).
//
//  Guarantees: an open, manifold input gives a closed, consistently wound output (rim = true); a closed input gives
//  the outer skin plus an inner cavity (no rim needed). Normals: layers keep the source corner normals (B negated),
//  rims are flat. Not guaranteed: no self-intersection test, so a thickness above the local feature size
//  (radius of curvature, gap to another sheet) can fold; non-manifold edges (3+ faces) get no rim.

import simd

public struct SolidifyModifier: Modifier {
    public var thickness: Double
    public var offset: Double          // -1…1
    public var rim: Bool
    public var evenThickness: Bool

    public init(thickness: Double, offset: Double = -1, rim: Bool = true, evenThickness: Bool = true) {
        self.thickness = thickness; self.offset = Swift.max(-1, Swift.min(1, offset))
        self.rim = rim; self.evenThickness = evenThickness
    }

    public var summary: String { "Solidify(t:\(thickness) offset:\(offset) rim:\(rim) even:\(evenThickness))" }

    public func apply(_ mesh: Mesh3) -> Mesh3 {
        guard !mesh.isEmpty, thickness != 0 else { return mesh }
        let w = WeldedMesh(mesh)
        // Even-thickness scale per welded vertex.
        var scale = [Double](repeating: 1, count: w.points.count)
        if evenThickness {
            var sum = [Double](repeating: 0, count: w.points.count), cnt = [Double](repeating: 0, count: w.points.count)
            for t in w.tris {
                let fn = cross3(w.points[t.y] - w.points[t.x], w.points[t.z] - w.points[t.x])
                guard len3(fn) > 1e-14 else { continue }
                let n = normalize3(fn)
                for k in [t.x, t.y, t.z] { sum[k] += dot3(n, w.normals[k]); cnt[k] += 1 }
            }
            for i in scale.indices where cnt[i] > 0 { scale[i] = 1 / Swift.max(0.25, sum[i] / cnt[i]) }
        }
        let aOff = thickness * (1 + offset) / 2, bOff = thickness * (offset - 1) / 2
        func layerA(_ id: Int) -> Vec3 { w.points[id] + w.normals[id] * (aOff * scale[id]) }
        func layerB(_ id: Int) -> Vec3 { w.points[id] + w.normals[id] * (bOff * scale[id]) }

        var out = Mesh3()
        func emit(_ p: (Vec3, Vec3, Vec3), n: (Vec3, Vec3, Vec3), uv: (Vec2, Vec2, Vec2)) {
            let base = UInt32(out.positions.count)
            out.positions += [p.0, p.1, p.2]; out.normals += [n.0, n.1, n.2]; out.uvs += [uv.0, uv.1, uv.2]
            out.indices += [base, base + 1, base + 2]
        }
        func flat(_ a: Vec3, _ b: Vec3, _ c: Vec3, uv: (Vec2, Vec2, Vec2) = (.zero, .zero, .zero)) {
            let n = cross3(b - a, c - a)
            guard len3(n) > 1e-14 else { return }
            let nn = normalize3(n)
            emit((a, b, c), n: (nn, nn, nn), uv: uv)
        }

        // Walk the original triangles by corner so UVs and normals carry across.
        var edgeCount: [SIMD2<Int>: Int] = [:]          // undirected welded edge -> uses
        var directed: [SIMD2<Int>: Int] = [:]           // directed edge -> uses (for the rim's direction)
        var t = 0
        while t + 2 < mesh.indices.count {
            let c = (Int(mesh.indices[t]), Int(mesh.indices[t + 1]), Int(mesh.indices[t + 2]))
            t += 3
            let id = (w.cornerID[c.0], w.cornerID[c.1], w.cornerID[c.2])
            guard id.0 != id.1, id.1 != id.2, id.0 != id.2 else { continue }
            let nrm: (Int) -> Vec3 = { mesh.normals.count == mesh.positions.count ? mesh.normals[$0] : Vec3(0, 1, 0) }
            let uv: (Int) -> Vec2 = { mesh.uvs.count == mesh.positions.count ? mesh.uvs[$0] : .zero }
            let a = (layerA(id.0), layerA(id.1), layerA(id.2))
            if mesh.normals.count == mesh.positions.count {
                emit(a, n: (nrm(c.0), nrm(c.1), nrm(c.2)), uv: (uv(c.0), uv(c.1), uv(c.2)))
                let b = (layerB(id.0), layerB(id.1), layerB(id.2))
                emit((b.0, b.2, b.1), n: (-nrm(c.0), -nrm(c.2), -nrm(c.1)), uv: (uv(c.0), uv(c.2), uv(c.1)))
            } else {
                flat(a.0, a.1, a.2, uv: (uv(c.0), uv(c.1), uv(c.2)))
                flat(layerB(id.0), layerB(id.2), layerB(id.1), uv: (uv(c.0), uv(c.2), uv(c.1)))
            }
            for (p, q) in [(id.0, id.1), (id.1, id.2), (id.2, id.0)] {
                edgeCount[SIMD2(Swift.min(p, q), Swift.max(p, q)), default: 0] += 1
                directed[SIMD2(p, q), default: 0] += 1
            }
        }
        guard rim else { return out }
        // Rim quads on every edge used by exactly one face (boundary), facing away from that face.
        for (e, n) in directed.sorted(by: { ($0.key.x, $0.key.y) < ($1.key.x, $1.key.y) }) where n == 1 && edgeCount[SIMD2(Swift.min(e.x, e.y), Swift.max(e.x, e.y))] == 1 {
            let (a, b) = (e.x, e.y)
            flat(layerA(a), layerB(a), layerB(b))
            flat(layerA(a), layerB(b), layerA(b))
        }
        return out
    }
}
