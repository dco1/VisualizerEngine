//  Shrinkwrap — snap a mesh's vertices onto another surface.
//
//   • nearestSurface — each vertex goes to the closest point on the target.
//   • project(…)     — each vertex is shot along its (welded, smooth) normal, its reverse, either (nearest hit), or a
//                      fixed direction, and lands on the first target triangle that ray meets. A vertex whose ray
//                      misses stays where it was.
//  `offset` then lifts the result along the target's surface normal at the landing point (positive = out of the
//  target's surface, by the target's winding); `factor` blends from the original position (0) to the wrapped one (1).
//
//  Guarantees: with offset 0 every wrapped vertex lies on the target within float rounding (not the faces between
//  them: a coarse mesh wraps a curved target only at its vertices); target normals are the target's own (interpolated
//  vertex normals if it has them). Not guaranteed: no edge-length or self-intersection protection, no "inside /
//  outside only" modes. The query structure is `TriangleBVH`. CPU only (see Modifier.swift).

import simd

public struct ShrinkwrapModifier: Modifier {
    public enum Sign: Sendable { case positive, negative, both }
    public enum Method: Sendable { case nearestSurface, projectAlongNormal(Sign), projectAlong(Vec3) }

    public var target: Mesh3
    public var method: Method
    public var offset: Double
    public var factor: Double
    private let bvh: TriangleBVH

    public init(target: Mesh3, method: Method = .nearestSurface, offset: Double = 0, factor: Double = 1) {
        self.target = target; self.method = method; self.offset = offset; self.factor = factor
        self.bvh = TriangleBVH(target)
    }

    public var summary: String {
        "Shrinkwrap(\(method) offset:\(offset) factor:\(factor) target:\(target.positions.count)v/\(target.triangleCount)t)"
    }

    public func apply(_ mesh: Mesh3) -> Mesh3 {
        guard !mesh.isEmpty, !target.isEmpty else { return mesh }
        let w = WeldedMesh(mesh)
        var newP = mesh.positions
        for i in newP.indices {
            let id = w.cornerID[i]
            let p = w.points[id]
            var landed: (Vec3, Vec3)? = nil                       // point, surface normal there
            switch method {
            case .nearestSurface:
                if let h = bvh.closest(to: p) { landed = (h.point, bvh.normal(tri: h.tri, bary: h.bary)) }
            case .projectAlongNormal(let sign):
                landed = cast(p, w.normals[id], sign)
            case .projectAlong(let d):
                landed = cast(p, normalize3(d), .both)
            }
            guard let (q, n) = landed else { continue }
            newP[i] = mesh.positions[i] + (q + n * offset - mesh.positions[i]) * factor
        }
        return mesh.moved(to: newP)
    }

    private func cast(_ p: Vec3, _ dir: Vec3, _ sign: Sign) -> (Vec3, Vec3)? {
        var best: TriangleBVH.Hit? = nil
        if sign != .negative, let h = bvh.raycast(origin: p, direction: dir) { best = h }
        if sign != .positive, let h = bvh.raycast(origin: p, direction: -dir), best == nil || h.t < best!.t { best = h }
        guard let h = best else { return nil }
        return (h.point, bvh.normal(tri: h.tri, bary: h.bary))
    }
}
