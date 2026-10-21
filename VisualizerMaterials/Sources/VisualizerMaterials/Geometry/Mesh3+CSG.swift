//  Solid booleans — union, subtracting, intersecting — for closed, outward-wound meshes.
//
//  Drill a hole, cut a slot, notch a tine, carve an engraving: every one is "this solid minus that solid", and
//  until now each was hand-built into the outline (or skipped). This is the classic BSP-tree CSG: each solid
//  becomes a binary space partition of its own faces, the faces of one are clipped to the inside/outside of the
//  other, and what survives is welded back into one soup.
//
//  What it guarantees and what it does not (read before trusting a result):
//   · Winding and normals are right: every output face is wound outward, normals interpolated across splits,
//     so `invertedWindingFaces == 0` and the volume is the volume of the boolean.
//   · It is NOT watertight by position. Splitting a face leaves T-junctions where a neighbour was not split at
//     the same point, so `MeshAudit.boundaryEdges` can be non-zero even though nothing is visibly open. Render it,
//     don't use it as a collision/closed-volume input without welding.
//   · Inputs must be CLOSED and outward-wound (`Mesh3.signedVolume > 0`). An open mesh has no inside.
//   · Coplanar faces (a cut that ends flush with a face) are the numerically fragile case: nudge the cutter 0.1 mm
//     past the face instead of ending it exactly on it.
//   · Cost grows with the product of the triangle counts; keep cutters coarse (a drill is 12–16 sides).
//   · Recursion depth follows the BSP depth: fine for hundreds to a few thousand triangles, not for 100 k.
//   · UVs are interpolated across splits, so a baked material keeps its mapping.

import simd

public extension Mesh3 {
    /// A ∪ B
    func union(_ other: Mesh3) -> Mesh3 { CSG.run(self, other, .union) }
    /// A − B (carve `other` out of `self`)
    func subtracting(_ other: Mesh3) -> Mesh3 { CSG.run(self, other, .subtract) }
    /// A ∩ B
    func intersection(_ other: Mesh3) -> Mesh3 { CSG.run(self, other, .intersect) }
}

private enum CSG {
    enum Op { case union, subtract, intersect }

    static let eps = 1e-7

    struct Vertex {
        var pos: Vec3, normal: Vec3, uv: Vec2
        func interpolated(to o: Vertex, _ t: Double) -> Vertex {
            Vertex(pos: pos + (o.pos - pos) * t, normal: normalize3(normal + (o.normal - normal) * t),
                   uv: uv + (o.uv - uv) * t)
        }
    }

    struct Plane {
        var normal: Vec3, w: Double
        init(normal: Vec3, w: Double) { self.normal = normal; self.w = w }
        init?(_ a: Vec3, _ b: Vec3, _ c: Vec3) {
            let n = cross3(b - a, c - a)
            guard len3(n) > 1e-14 else { return nil }
            let nn = normalize3(n)
            self.init(normal: nn, w: dot3(nn, a))
        }
        mutating func flip() { normal = -normal; w = -w }
    }

    struct Polygon {
        var verts: [Vertex]
        var plane: Plane
        mutating func flip() {
            verts.reverse()
            for i in verts.indices { verts[i].normal = -verts[i].normal }
            plane.flip()
        }
    }

    struct Split { var coplanarFront: [Polygon] = [], coplanarBack: [Polygon] = [], front: [Polygon] = [], back: [Polygon] = [] }

    /// Split `poly` by `plane` into the four buckets, appended to `out`.
    static func split(_ poly: Polygon, by plane: Plane, into out: inout Split) {
        let coplanar = 0, fr = 1, bk = 2, spanning = 3
        var polyType = 0
        var types: [Int] = []
        types.reserveCapacity(poly.verts.count)
        for v in poly.verts {
            let t = dot3(plane.normal, v.pos) - plane.w
            let type = t < -eps ? bk : (t > eps ? fr : coplanar)
            polyType |= type
            types.append(type)
        }
        switch polyType {
        case coplanar:
            if dot3(plane.normal, poly.plane.normal) > 0 { out.coplanarFront.append(poly) } else { out.coplanarBack.append(poly) }
        case fr: out.front.append(poly)
        case bk: out.back.append(poly)
        default:
            _ = spanning
            var f: [Vertex] = [], b: [Vertex] = []
            let n = poly.verts.count
            for i in 0..<n {
                let j = (i + 1) % n
                let ti = types[i], tj = types[j]
                let vi = poly.verts[i], vj = poly.verts[j]
                if ti != bk { f.append(vi) }
                if ti != fr { b.append(vi) }
                if (ti | tj) == spanning {
                    let t = (plane.w - dot3(plane.normal, vi.pos)) / dot3(plane.normal, vj.pos - vi.pos)
                    let v = vi.interpolated(to: vj, t)
                    f.append(v); b.append(v)
                }
            }
            if f.count >= 3 { out.front.append(Polygon(verts: f, plane: poly.plane)) }
            if b.count >= 3 { out.back.append(Polygon(verts: b, plane: poly.plane)) }
        }
    }

    final class Node {
        var plane: Plane?
        var front: Node?
        var back: Node?
        var polygons: [Polygon] = []

        init(_ polys: [Polygon] = []) { if !polys.isEmpty { build(polys) } }

        func invert() {
            for i in polygons.indices { polygons[i].flip() }
            plane?.flip()
            front?.invert(); back?.invert()
            swap(&front, &back)
        }

        func clip(_ polys: [Polygon]) -> [Polygon] {
            guard let plane else { return polys }
            var sp = Split()
            for p in polys { split(p, by: plane, into: &sp) }
            // coplanar faces go with the side they face, exactly as csg.js buckets them
            let f = front?.clip(sp.coplanarFront + sp.front) ?? (sp.coplanarFront + sp.front)
            let b = back?.clip(sp.coplanarBack + sp.back) ?? []     // no back child: what is behind is inside → dropped
            return f + b
        }

        func clipTo(_ other: Node) {
            polygons = other.clip(polygons)
            front?.clipTo(other); back?.clipTo(other)
        }

        func all() -> [Polygon] { polygons + (front?.all() ?? []) + (back?.all() ?? []) }

        func build(_ polys: [Polygon]) {
            guard !polys.isEmpty else { return }
            if plane == nil { plane = polys[0].plane }
            var sp = Split()
            for p in polys { split(p, by: plane!, into: &sp) }
            polygons += sp.coplanarFront + sp.coplanarBack
            if !sp.front.isEmpty { if front == nil { front = Node() }; front!.build(sp.front) }
            if !sp.back.isEmpty { if back == nil { back = Node() }; back!.build(sp.back) }
        }
    }

    static func polygons(of m: Mesh3) -> [Polygon] {
        var out: [Polygon] = []
        var t = 0
        while t + 2 < m.indices.count {
            let i = [Int(m.indices[t]), Int(m.indices[t + 1]), Int(m.indices[t + 2])]
            t += 3
            guard let pl = Plane(m.positions[i[0]], m.positions[i[1]], m.positions[i[2]]) else { continue }
            let verts = i.map { k in
                Vertex(pos: m.positions[k], normal: m.normals.count > k ? m.normals[k] : pl.normal,
                       uv: m.uvs.count > k ? m.uvs[k] : .zero)
            }
            out.append(Polygon(verts: verts, plane: pl))
        }
        return out
    }

    static func mesh(from polys: [Polygon]) -> Mesh3 {
        var out = Mesh3()
        for p in polys where p.verts.count >= 3 {
            for i in 1..<(p.verts.count - 1) {
                let tri = [p.verts[0], p.verts[i], p.verts[i + 1]]
                // A sliver left by a split can be degenerate; the plane normal still says which way is out.
                guard len3(cross3(tri[1].pos - tri[0].pos, tri[2].pos - tri[0].pos)) > 1e-14 else { continue }
                let base = UInt32(out.positions.count)
                for v in tri { out.positions.append(v.pos); out.normals.append(v.normal); out.uvs.append(v.uv) }
                out.indices.append(contentsOf: [base, base + 1, base + 2])
            }
        }
        return out
    }

    /// Bounding box of a polygon list, grown by `pad`.
    static func box(of polys: [Polygon], pad: Double) -> (lo: Vec3, hi: Vec3) {
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for p in polys { for v in p.verts { lo = simd_min(lo, v.pos); hi = simd_max(hi, v.pos) } }
        return (lo - Vec3(repeating: pad), hi + Vec3(repeating: pad))
    }

    static func touches(_ p: Polygon, _ lo: Vec3, _ hi: Vec3) -> Bool {
        var plo = Vec3(repeating: .infinity), phi = Vec3(repeating: -.infinity)
        for v in p.verts { plo = simd_min(plo, v.pos); phi = simd_max(phi, v.pos) }
        return plo.x <= hi.x && phi.x >= lo.x && plo.y <= hi.y && phi.y >= lo.y && plo.z <= hi.z && phi.z >= lo.z
    }

    /// LOCALISED: only the polygons of A whose bounding box touches B's are cut. The rest of A cannot meet B, so it is
    /// outside it — kept untouched for a union or subtraction, dropped for an intersection. Classic BSP CSG instead cuts
    /// EVERY polygon of A along every plane of B's tree, so two slots in a 1 300-triangle toaster body made it 9 000
    /// (DH-1448): the planes of a small cutter, extended to infinity, slice the whole body into slivers.
    ///
    /// B's own polygons still have to be classified against ALL of A — which side of the solid a point is on depends on
    /// the whole surface — so a second tree of the complete A (`full`) is used for that, and kept in step with the
    /// near-only tree through the same inversions. The classification only reads a tree's planes and shape, not its
    /// polygon lists, so the polygons that `clipTo` removes from the near tree do not change what `full` says.
    static func run(_ a: Mesh3, _ b: Mesh3, _ op: Op) -> Mesh3 {
        let pa = polygons(of: a), pb = polygons(of: b)
        guard !pa.isEmpty else { return op == .union ? b : Mesh3() }
        guard !pb.isEmpty else { return op == .intersect ? Mesh3() : a }
        let bb = box(of: pb, pad: 1e-6)
        var near: [Polygon] = [], far: [Polygon] = []
        for p in pa { if touches(p, bb.lo, bb.hi) { near.append(p) } else { far.append(p) } }

        let nb = Node(pb)
        let full = Node(pa)             // classifies B's polygons: needs the whole of A
        let nn = Node(near)             // the part of A that is actually cut
        switch op {
        case .union:
            nn.clipTo(nb); nb.clipTo(full); nb.invert(); nb.clipTo(full); nb.invert()
            nn.build(nb.all())
            return mesh(from: far + nn.all())
        case .subtract:
            nn.invert(); full.invert()
            nn.clipTo(nb); nb.clipTo(full); nb.invert(); nb.clipTo(full); nb.invert()
            nn.build(nb.all()); nn.invert()
            return mesh(from: far + nn.all())
        case .intersect:
            nn.invert(); full.invert()
            nb.clipTo(full); nb.invert(); nn.clipTo(nb); nb.clipTo(full)
            nn.build(nb.all()); nn.invert()
            return mesh(from: nn.all())                  // A's far polygons are not in the intersection
        }
    }
}
