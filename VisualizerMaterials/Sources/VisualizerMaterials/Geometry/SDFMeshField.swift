//  SDFMeshField.swift — mesh-to-SDF: exact signed distance to a closed triangle mesh (ticket item C).
//
//  Distance: closest point on the nearest triangle, found through a BVH (median split, best-first pruning).
//  Sign: the angle-weighted pseudo-normal of the closest feature (face / edge / vertex), Bærentzen & Aanæs —
//  exact for watertight, consistently wound meshes with no winding-number sum over every triangle.
//  The mesh is welded by position first, so flat-shaded `Mesh3`s (three vertices a triangle) work.
//  Not guaranteed: open or self-intersecting meshes give the right |distance| and a sign that can be wrong
//  near the defect; inside-out meshes give an inverted field.

import simd

public final class SDFMeshField: Sendable {
    private struct Tri: Sendable { var a: Int32, b: Int32, c: Int32; var n: Vec3 }
    private struct BNode: Sendable { var lo: Vec3, hi: Vec3; var left: Int32, right: Int32, start: Int32, count: Int32 }

    private let verts: [Vec3]
    private let tris: [Tri]
    private let vertexNormals: [Vec3]
    private let edgeNormals: [UInt64: Vec3]
    private let nodes: [BNode]
    private let order: [Int32]

    private static func edgeKey(_ a: Int32, _ b: Int32) -> UInt64 {
        let lo = UInt64(UInt32(Swift.min(a, b))), hi = UInt64(UInt32(Swift.max(a, b)))
        return lo << 32 | hi
    }

    public init(_ mesh: Mesh3) {
        // Weld by position (1e-9 m grid).
        struct Key: Hashable { let x: Int64, y: Int64, z: Int64 }
        var map: [Key: Int32] = [:]
        var verts: [Vec3] = []
        var remap = [Int32](repeating: 0, count: mesh.positions.count)
        for (i, p) in mesh.positions.enumerated() {
            let k = Key(x: Int64((p.x * 1e9).rounded()), y: Int64((p.y * 1e9).rounded()), z: Int64((p.z * 1e9).rounded()))
            if let j = map[k] { remap[i] = j } else {
                let j = Int32(verts.count); map[k] = j; verts.append(p); remap[i] = j
            }
        }
        var tris: [Tri] = []
        var vn = [Vec3](repeating: .zero, count: verts.count)
        var en: [UInt64: Vec3] = [:]
        var t = 0
        while t + 2 < mesh.indices.count {
            let a = remap[Int(mesh.indices[t])], b = remap[Int(mesh.indices[t + 1])], c = remap[Int(mesh.indices[t + 2])]
            t += 3
            guard a != b, b != c, a != c else { continue }
            let pa = verts[Int(a)], pb = verts[Int(b)], pc = verts[Int(c)]
            let raw = cross3(pb - pa, pc - pa)
            let area2 = len3(raw)
            guard area2 > 1e-18 else { continue }
            let n = raw / area2
            tris.append(Tri(a: a, b: b, c: c, n: n))
            // Angle-weighted vertex normals; edge normals are the sum of the two faces.
            func angle(_ p: Vec3, _ q: Vec3, _ r: Vec3) -> Double {
                let u = normalize3(q - p), v = normalize3(r - p)
                return acos(Swift.max(-1, Swift.min(1, dot3(u, v))))
            }
            vn[Int(a)] += n * angle(pa, pb, pc)
            vn[Int(b)] += n * angle(pb, pc, pa)
            vn[Int(c)] += n * angle(pc, pa, pb)
            en[SDFMeshField.edgeKey(a, b), default: .zero] += n
            en[SDFMeshField.edgeKey(b, c), default: .zero] += n
            en[SDFMeshField.edgeKey(c, a), default: .zero] += n
        }
        self.verts = verts; self.tris = tris; self.vertexNormals = vn; self.edgeNormals = en

        // BVH.
        var idx = (0..<tris.count).map { Int32($0) }
        var nodes: [BNode] = []
        func box(_ i: Int32) -> (Vec3, Vec3) {
            let tr = tris[Int(i)]
            let a = verts[Int(tr.a)], b = verts[Int(tr.b)], c = verts[Int(tr.c)]
            return (simd_min(a, simd_min(b, c)), simd_max(a, simd_max(b, c)))
        }
        func build(_ lo: Int, _ hi: Int) -> Int32 {
            var bl = Vec3(repeating: .infinity), bh = Vec3(repeating: -.infinity)
            var cl = Vec3(repeating: .infinity), ch = Vec3(repeating: -.infinity)
            for i in lo..<hi {
                let (l, h) = box(idx[i]); bl = simd_min(bl, l); bh = simd_max(bh, h)
                let c = (l + h) * 0.5; cl = simd_min(cl, c); ch = simd_max(ch, c)
            }
            let me = Int32(nodes.count)
            nodes.append(BNode(lo: bl, hi: bh, left: -1, right: -1, start: Int32(lo), count: Int32(hi - lo)))
            if hi - lo <= 4 { return me }
            let ext = ch - cl
            let axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : (ext.y >= ext.z ? 1 : 2)
            guard ext[axis] > 0 else { return me }
            idx[lo..<hi].sort { (box($0).0[axis] + box($0).1[axis]) < (box($1).0[axis] + box($1).1[axis]) }
            let mid = (lo + hi) / 2
            let l = build(lo, mid), r = build(mid, hi)
            nodes[Int(me)].left = l; nodes[Int(me)].right = r; nodes[Int(me)].count = 0
            return me
        }
        if !tris.isEmpty { _ = build(0, tris.count) }
        self.nodes = nodes; self.order = idx
    }

    public var triangleCount: Int { tris.count }

    private static func boxDist2(_ n: BNode, _ p: Vec3) -> Double {
        let d = simd_max(simd_max(n.lo - p, p - n.hi), .zero)
        return dot3(d, d)
    }

    /// Closest point on triangle (Ericson) and which feature it is on: 0 face, 1/2/3 vertex a/b/c, 4 ab, 5 bc, 6 ca.
    private static func closest(_ p: Vec3, _ a: Vec3, _ b: Vec3, _ c: Vec3) -> (Vec3, Int) {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = dot3(ab, ap), d2 = dot3(ac, ap)
        if d1 <= 0 && d2 <= 0 { return (a, 1) }
        let bp = p - b
        let d3 = dot3(ab, bp), d4 = dot3(ac, bp)
        if d3 >= 0 && d4 <= d3 { return (b, 2) }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0 && d1 >= 0 && d3 <= 0 { return (a + ab * (d1 / (d1 - d3)), 4) }
        let cp = p - c
        let d5 = dot3(ab, cp), d6 = dot3(ac, cp)
        if d6 >= 0 && d5 <= d6 { return (c, 3) }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0 && d2 >= 0 && d6 <= 0 { return (a + ac * (d2 / (d2 - d6)), 6) }
        let va = d3 * d6 - d5 * d4
        if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return (b + (c - b) * w, 5)
        }
        let denom = 1 / (va + vb + vc)
        return (a + ab * (vb * denom) + ac * (vc * denom), 0)
    }

    /// Unsigned distance to the surface.
    public func unsignedDistance(_ p: Vec3) -> Double { query(p).distance }

    public func signedDistance(_ p: Vec3) -> Double {
        let q = query(p)
        return q.inside ? -q.distance : q.distance
    }

    private func query(_ p: Vec3) -> (distance: Double, inside: Bool) {
        guard !nodes.isEmpty else { return (.infinity, false) }
        var best = Double.infinity, bestPoint = Vec3.zero, bestTri = -1, bestFeature = 0
        var stack: [Int32] = [0]
        stack.reserveCapacity(64)
        while let ni = stack.popLast() {
            let n = nodes[Int(ni)]
            if SDFMeshField.boxDist2(n, p) >= best { continue }
            if n.left < 0 {
                for i in Int(n.start)..<Int(n.start + n.count) {
                    let ti = Int(order[i]), t = tris[ti]
                    let (c, f) = SDFMeshField.closest(p, verts[Int(t.a)], verts[Int(t.b)], verts[Int(t.c)])
                    let d = p - c, d2 = dot3(d, d)
                    if d2 < best { best = d2; bestPoint = c; bestTri = ti; bestFeature = f }
                }
            } else {
                let l = nodes[Int(n.left)], r = nodes[Int(n.right)]
                if SDFMeshField.boxDist2(l, p) < SDFMeshField.boxDist2(r, p) {
                    stack.append(n.right); stack.append(n.left)
                } else {
                    stack.append(n.left); stack.append(n.right)
                }
            }
        }
        guard bestTri >= 0 else { return (.infinity, false) }
        let t = tris[bestTri]
        let normal: Vec3
        switch bestFeature {
        case 1: normal = vertexNormals[Int(t.a)]
        case 2: normal = vertexNormals[Int(t.b)]
        case 3: normal = vertexNormals[Int(t.c)]
        case 4: normal = edgeNormals[SDFMeshField.edgeKey(t.a, t.b)] ?? t.n
        case 5: normal = edgeNormals[SDFMeshField.edgeKey(t.b, t.c)] ?? t.n
        case 6: normal = edgeNormals[SDFMeshField.edgeKey(t.c, t.a)] ?? t.n
        default: normal = t.n
        }
        let dist = best.squareRoot()
        let inside = dist > 0 ? dot3(p - bestPoint, normal) < 0 : false
        return (dist, inside)
    }
}
