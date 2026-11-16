//  HalfEdge+Inset — inset faces: a region (one inset border, the faces inside stay joined) or each face on its
//  own, with an optional depth (the inset faces sink or rise along their normal).
//
//  Every vertex on the region's border is duplicated `thickness` inside (mitred in the face's plane, measured
//  square to each edge, clamped so it cannot pass the faces' centres); the region's faces are re-added on the
//  copies (same face ids) and one quad ring face stands on every border edge. Region vertices that are not on the
//  border only move by `depth`. Area is conserved by a flat inset: inner + ring = the original.
//  Refuses (nil, unchanged) when a border vertex is pinched (the region touches itself there).

import simd

public extension HalfEdgeMesh {

    /// Returns the inset (inner) faces.
    @discardableResult
    mutating func inset(_ sel: HESelection, thickness: Double, depth: Double = 0, individual: Bool = false) -> HESelection? {
        let region = selectedFaces(sel)
        guard !region.isEmpty, thickness >= 0 else { return nil }
        let backup = self
        var out: [Int] = []
        for g in (individual ? region.map { [$0] } : [region]) {
            guard let ids = insetRegion(g, thickness: thickness, depth: depth) else { self = backup; return nil }
            out += ids
        }
        return selection(.face, out)
    }

    private mutating func insetRegion(_ region: [Int], thickness t: Double, depth: Double) -> [Int]? {
        let inRegion = Set(region)
        var outH: [Int: Int] = [:], inH: [Int: Int] = [:], border: [Int] = []
        for f in region { for h in halfedges(of: f) where heTwin[h] < 0 || !inRegion.contains(heFace[heTwin[h]]) {
            let a = heOrigin[h], b = heOrigin[heNext[h]]
            guard outH.updateValue(h, forKey: a) == nil, inH.updateValue(h, forKey: b) == nil else { return nil }
            border.append(h)
        } }
        // unit normal and centre distance per region vertex
        var nv: [Int: Vec3] = [:], near: [Int: Double] = [:]
        for f in region {
            let av = areaVector(f), c = faceCentroid(f)
            for v in vertices(of: f) { nv[v, default: .zero] += av; near[v] = Swift.min(near[v] ?? .infinity, dist3(c, positions[v])) }
        }
        var newPos: [Int: Vec3] = [:]
        for (v, a) in nv { nv[v] = normalize3(a); newPos[v] = positions[v] + nv[v]! * depth }
        for (v, ho) in outH {
            let hi = inH[v]!, n = nv[v]!
            let p = positions[v], prev = positions[heOrigin[hi]], next = positions[heOrigin[heNext[ho]]]
            let a = normalize3(cross3(n, normalize3(p - prev))), b = normalize3(cross3(n, normalize3(next - p)))
            let k = 1 + dot3(a, b)
            var w = k > 1e-6 ? (a + b) * (t / k) : a * t
            let cap = 0.9 * (near[v] ?? .infinity)
            if len3(w) > cap { w *= cap / len3(w) }
            newPos[v] = p + w + n * depth
        }
        var dup: [Int: Int] = [:]
        for v in outH.keys.sorted() { dup[v] = addVertex(newPos[v]!) }
        for (v, _) in nv where dup[v] == nil { positions[v] = newPos[v]! }     // interior vertices: only the depth
        var adds: [EditPoly] = []
        for f in region { var p = editPoly(of: f); p.v = p.v.map { dup[$0] ?? $0 }; adds.append(p) }
        for h in border {
            let a = heOrigin[h], b = heOrigin[heNext[h]], ua = heUV[h], ub = heUV[heNext[h]]
            adds.append(EditPoly(v: [a, b, dup[b]!, dup[a]!], uv: [ua, ub, ub, ua]))
        }
        guard let ids = rebuild(removing: region, adding: adds) else { return nil }
        return Array(ids.prefix(region.count))
    }
}
