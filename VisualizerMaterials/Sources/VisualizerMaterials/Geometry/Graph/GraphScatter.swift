//  GraphScatter.swift — `distribute` (points on a surface) and `instance` (a mesh on every point).
//
//  distribute: `count` points, uniform by AREA over the mesh's triangles (cumulative-area search + uniform
//  barycentrics), each with its face normal; driven by a SplitMix64 seeded from `seed`, so the same seed and mesh
//  give the same points on every run and platform. instance: a copy of `mesh` at each point, scaled by a random
//  factor in `scaleMin…scaleMax`, optionally turned so +Y follows the point's normal (`alignToNormal`), then
//  jittered by up to `rotateJitter` degrees about each axis (vec3), seeded by `seed`.
//
//  Guarantees: deterministic, and the instance count is exactly the point count. Not guaranteed: no relaxation
//  (points may cluster; it is white noise), and instances may overlap each other.
//  CPU: it is O(points × instance size) memory traffic, dominated by the output copy, not arithmetic.

import Foundation
import simd

enum GraphScatter {
    static func compute(_ n: GraphNode, _ a: GraphArgs, _ ins: [String: GraphValue]) throws -> GraphValue {
        switch n.kind {
        case .distribute:
            return .points(distribute(try GraphEvaluator.mesh(n, ins, "mesh"), count: a.int("count", 100), seed: UInt64(Swift.max(0, a.int("seed", 0)))))
        default:
            guard let pts = ins["points"]?.points else { throw GraphError.missingInput(node: n.id, kind: n.kind.rawValue, input: "points") }
            return .mesh(instance(try GraphEvaluator.mesh(n, ins, "mesh"), at: pts, a))
        }
    }

    static func distribute(_ m: Mesh3, count: Int, seed: UInt64) -> [GraphPoint] {
        var cum: [Double] = [], tris: [(Vec3, Vec3, Vec3)] = []
        var total = 0.0, i = 0
        while i + 2 < m.indices.count {
            let p = (m.positions[Int(m.indices[i])], m.positions[Int(m.indices[i + 1])], m.positions[Int(m.indices[i + 2])])
            i += 3
            let area = len3(cross3(p.1 - p.0, p.2 - p.0)) / 2
            guard area > 1e-14 else { continue }
            total += area; cum.append(total); tris.append(p)
        }
        guard total > 0, count > 0 else { return [] }
        var rng = SplitMix64(seed: seed)
        return (0 ..< count).map { _ in
            let target = rng.nextUnit() * total
            var lo = 0, hi = cum.count - 1
            while lo < hi { let mid = (lo + hi) / 2; if cum[mid] < target { lo = mid + 1 } else { hi = mid } }
            let t = tris[lo]
            let r1 = rng.nextUnit().squareRoot(), r2 = rng.nextUnit()
            let p = t.0 * (1 - r1) + t.1 * (r1 * (1 - r2)) + t.2 * (r1 * r2)
            return GraphPoint(position: p, normal: normalize3(cross3(t.1 - t.0, t.2 - t.0)))
        }
    }

    static func instance(_ m: Mesh3, at pts: [GraphPoint], _ a: GraphArgs) -> Mesh3 {
        let s0 = a.double("scaleMin", 1), s1 = a.double("scaleMax", s0)
        let jitter = a.vec3("rotateJitter", .zero) * (.pi / 180)
        let align = a.bool("alignToNormal", false)
        var rng = SplitMix64(seed: UInt64(Swift.max(0, a.int("seed", 0))) &+ 0x5EED)
        var out = Mesh3()
        out.positions.reserveCapacity(pts.count * m.positions.count)
        out.normals.reserveCapacity(pts.count * m.normals.count)
        out.uvs.reserveCapacity(pts.count * m.uvs.count)
        out.indices.reserveCapacity(pts.count * m.indices.count)
        for p in pts {
            let s = s0 + (s1 - s0) * rng.nextUnit()
            let jx = (rng.nextUnit() * 2 - 1) * jitter.x, jy = (rng.nextUnit() * 2 - 1) * jitter.y, jz = (rng.nextUnit() * 2 - 1) * jitter.z
            var q = simd_quatd(angle: jz, axis: Vec3(0, 0, 1)) * simd_quatd(angle: jy, axis: Vec3(0, 1, 0)) * simd_quatd(angle: jx, axis: Vec3(1, 0, 0))
            if align { q = simd_quatd(from: Vec3(0, 1, 0), to: p.normal) * q }
            out.append(m.placed(scale: Vec3(repeating: s), rotation: q, translation: p.position))
        }
        return out
    }
}
