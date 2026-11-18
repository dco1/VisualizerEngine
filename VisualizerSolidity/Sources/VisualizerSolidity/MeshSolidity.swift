//  Mesh3 → auditors. See Package.swift for why this is its own package.
//
//  `closedSolidFailures` — every mesh is a closed manifold solid (MeshSoundness, after welding).
//  `zFightFailures`      — no two same-facing faces of different meshes share a depth (ZFightDetector).
//  Both return human-readable failure strings (empty = clean), so any test framework can assert on them.

import simd
import VisualizerCore
import VisualizerMaterials

public enum MeshSolidity {

    /// Mesh3 builders emit unwelded quads; weld by quantised position (1e-5) so shared edges count.
    public static func welded(_ m: Mesh3) -> (positions: [SIMD3<Float>], indices: [UInt32]) {
        var map: [SIMD3<Int64>: UInt32] = [:], pos: [SIMD3<Float>] = []
        let idx: [UInt32] = m.indices.map { i in
            let p = m.positions[Int(i)]
            let key = SIMD3<Int64>(Int64((p.x * 1e5).rounded()), Int64((p.y * 1e5).rounded()), Int64((p.z * 1e5).rounded()))
            if let j = map[key] { return j }
            let j = UInt32(pos.count); map[key] = j; pos.append(SIMD3(Float(p.x), Float(p.y), Float(p.z)))
            return j
        }
        return (pos, idx)
    }

    /// One failure line per mesh that is not a closed solid, labelled `label#index`.
    public static func closedSolidFailures(_ meshes: [Mesh3], label: String) -> [String] {
        meshes.enumerated().compactMap { n, m in
            let w = welded(m)
            let r = MeshSoundness.audit(positions: w.positions, indices: w.indices, label: "\(label)#\(n)", expectClosed: true)
            return r.isSound ? nil : "\(label) part \(n) is not a closed solid: \(r.summary)"
        }
    }

    /// Can an orthographic camera `cameraPitchDegrees` down, at any yaw, ever see a face with this normal?
    /// Faces it can't (undersides) can't show z-fighting.
    public static func everVisible(_ n: SIMD3<Float>, cameraPitchDegrees: Float) -> Bool {
        let p = cameraPitchDegrees * .pi / 180
        return cosf(p) * simd_length(SIMD2(n.x, n.z)) + sinf(p) * n.y > 1e-4
    }

    /// Faces of one object (one mesh per part), as patches. Quads (two triangles sharing a diagonal) merge
    /// back into exact rectangles; faces the camera can never see are left out; faces of the same part
    /// never count against each other (`owner` = part index).
    public static func patches(_ meshes: [Mesh3], name: String, cameraPitchDegrees: Float) -> [SurfacePatch] {
        var out: [SurfacePatch] = []
        let f = { (v: Vec3) in SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)) }
        for (pi, m) in meshes.enumerated() {
            var tris: [[SIMD3<Float>]] = stride(from: 0, to: m.indices.count, by: 3).map { t in
                (0..<3).map { f(m.positions[Int(m.indices[t + $0])]) }
            }
            while !tris.isEmpty {
                let t = tris.removeFirst()
                let n = simd_normalize(simd_cross(t[1] - t[0], t[2] - t[0]))
                var pts = t
                // Its quad partner: the next triangle with the same normal sharing two corners.
                if let k = tris.firstIndex(where: { o in
                    let on = simd_normalize(simd_cross(o[1] - o[0], o[2] - o[0]))
                    return simd_dot(on, n) > 0.9999 && o.filter { p in t.contains { simd_distance($0, p) < 1e-6 } }.count == 2
                }) {
                    let o = tris.remove(at: k)
                    pts += o.filter { p in !t.contains { simd_distance($0, p) < 1e-6 } }
                }
                if !everVisible(n, cameraPitchDegrees: cameraPitchDegrees) { continue }
                out.append(patch(pts, normal: n, name: "\(name)#\(pi)", owner: pi))
            }
        }
        return out
    }

    /// A planar polygon → patch: its exact in-plane bounding rectangle (quads: along a side, not the
    /// diagonal; triangles: along the longest edge).
    public static func patch(_ pts: [SIMD3<Float>], normal n: SIMD3<Float>, name: String, owner: Int?) -> SurfacePatch {
        let a = pts[0]
        let u: SIMD3<Float>
        if pts.count == 3 {
            let edges = [(pts[1] - pts[0]), (pts[2] - pts[1]), (pts[0] - pts[2])]
            u = simd_normalize(edges.max { simd_length($0) < simd_length($1) }!)
        } else {
            let others = pts.dropFirst().sorted { simd_distance($0, a) < simd_distance($1, a) }
            u = simd_normalize(others[0] - a)
        }
        let v = simd_cross(n, u)
        let us = pts.map { simd_dot($0 - a, u) }, vs = pts.map { simd_dot($0 - a, v) }
        let centre = a + u * (us.min()! + us.max()!) / 2 + v * (vs.min()! + vs.max()!) / 2
        let hu = (us.max()! - us.min()!) / 2, hv = (vs.max()! - vs.min()!) / 2
        return SurfacePatch(name: name, center: centre, normal: n, radius: max(hu, hv), halfU: u * hu, halfV: v * hv, owner: owner)
    }

    public static func patch(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, name: String, owner: Int?) -> SurfacePatch {
        patch([a, b, c], normal: simd_normalize(simd_cross(b - a, c - a)), name: name, owner: owner)
    }

    /// One failure line (the worst fighting pairs) when two same-facing faces of different parts share a depth.
    public static func zFightFailures(_ meshes: [Mesh3], label: String, cameraPitchDegrees: Float,
                                      overlapMargin: Float = 0.002) -> [String] {
        let r = ZFightDetector.audit(patches(meshes, name: label, cameraPitchDegrees: cameraPitchDegrees),
                                     requireSameFacing: true, overlapMargin: overlapMargin)
        return r.isClean ? [] : ["\(label): \(r.worst.map { "\($0.a) vs \($0.b) Δ\($0.separation)" })"]
    }
}
