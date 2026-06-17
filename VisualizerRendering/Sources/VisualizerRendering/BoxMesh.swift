import Foundation
import simd

// ── BoxMesh ───────────────────────────────────────────────────────────────────
//
// A UNIT cube ([-0.5, 0.5]³) in the instanced-renderer mesh format (`CoinMesh.Mesh`).
// Box rigid bodies bake their per-instance half-extents into the `CoinTransform`
// basis (see `coinDeriveTransforms` in CoinDEM.metal), so the mesh is unit-sized and
// every die / crate / block comes out at its true size from ONE shared mesh.
//
// Six faces, each with its own outward normal so the cube edges stay sharp, wound
// CCW-from-outside to match `IlluminatoramaMesh.unitBox` (the renderer's front-face
// convention) — so the cube is solid, not inside-out.

public enum BoxMesh {

    /// A unit cube centred at the origin, side length 1 (half-extent 0.5).
    public static func unit() -> CoinMesh.Mesh {
        let h: Float = 0.5
        // (face normal, 4 corner positions CCW seen from outside).
        let faces: [(SIMD3<Float>, [SIMD3<Float>])] = [
            (SIMD3( 1, 0, 0), [SIMD3( h,-h, h), SIMD3( h,-h,-h), SIMD3( h, h,-h), SIMD3( h, h, h)]),
            (SIMD3(-1, 0, 0), [SIMD3(-h,-h,-h), SIMD3(-h,-h, h), SIMD3(-h, h, h), SIMD3(-h, h,-h)]),
            (SIMD3( 0, 1, 0), [SIMD3(-h, h, h), SIMD3( h, h, h), SIMD3( h, h,-h), SIMD3(-h, h,-h)]),
            (SIMD3( 0,-1, 0), [SIMD3(-h,-h,-h), SIMD3( h,-h,-h), SIMD3( h,-h, h), SIMD3(-h,-h, h)]),
            (SIMD3( 0, 0, 1), [SIMD3(-h,-h, h), SIMD3( h,-h, h), SIMD3( h, h, h), SIMD3(-h, h, h)]),
            (SIMD3( 0, 0,-1), [SIMD3( h,-h,-h), SIMD3(-h,-h,-h), SIMD3(-h, h,-h), SIMD3( h, h,-h)]),
        ]
        var pos: [SIMD3<Float>] = [], nrm: [SIMD3<Float>] = []
        var uv:  [SIMD2<Float>] = [], idx: [Int32] = []
        for (normal, quad) in faces {
            let base = Int32(pos.count)
            for (i, p) in quad.enumerated() {
                pos.append(p); nrm.append(normal)
                uv.append(SIMD2(Float(i & 1), Float((i >> 1) & 1)))
            }
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        }
        return CoinMesh.Mesh(positions: pos, normals: nrm, uvs: uv, indices: idx)
    }

    /// A CHAMFERED (bevel-edged) unit cube — real objects never have perfectly sharp
    /// 90° edges, so a bevel kills the CG-plastic tell on dice/blocks. `chamfer` is the
    /// edge cut as a fraction of the half-extent (0.5). Built as 6 inset face quads +
    /// 12 edge bevels + 8 corner facets, per-face flat normals, winding auto-corrected
    /// so nothing renders inside-out. Still a unit cube ([-0.5,0.5]³): box bodies bake
    /// their per-instance extents into the transform, so it sizes per die.
    public static func unitBeveled(chamfer: Float = 0.10) -> CoinMesh.Mesh {
        let h: Float = 0.5
        let a = h - max(0.01, min(0.45, chamfer)) * h     // inset distance
        var pos: [SIMD3<Float>] = [], nrm: [SIMD3<Float>] = []
        var uv: [SIMD2<Float>] = [], idx: [Int32] = []

        func emit(_ ps: [SIMD3<Float>], _ outward: SIMD3<Float>) {
            // Ensure CCW-from-outside: if the geometric normal opposes `outward`, reverse.
            var q = ps
            let gn = simd_cross(q[1] - q[0], q[2] - q[0])
            if simd_dot(gn, outward) < 0 { q.reverse() }
            let base = Int32(pos.count); let n = simd_normalize(outward)
            for p in q { pos.append(p); nrm.append(n); uv.append(SIMD2(0, 0)) }
            if q.count == 4 { idx += [base, base+1, base+2, base, base+2, base+3] }
            else { idx += [base, base+1, base+2] }
        }

        let signs: [Float] = [-1, 1]
        // 6 inset face quads.
        for s in signs {
            emit([SIMD3(s*h, -a, -a), SIMD3(s*h, a, -a), SIMD3(s*h, a, a), SIMD3(s*h, -a, a)], SIMD3(s, 0, 0))
            emit([SIMD3(-a, s*h, -a), SIMD3(a, s*h, -a), SIMD3(a, s*h, a), SIMD3(-a, s*h, a)], SIMD3(0, s, 0))
            emit([SIMD3(-a, -a, s*h), SIMD3(a, -a, s*h), SIMD3(a, a, s*h), SIMD3(-a, a, s*h)], SIMD3(0, 0, s))
        }
        // 12 edge bevels (4 per axis the edge runs along).
        for u in signs { for v in signs {
            // edge ∥ Z (between ±X and ±Y faces)
            emit([SIMD3(u*h, v*a, -a), SIMD3(u*h, v*a, a), SIMD3(u*a, v*h, a), SIMD3(u*a, v*h, -a)], SIMD3(u, v, 0))
            // edge ∥ Y (between ±X and ±Z faces)
            emit([SIMD3(u*h, -a, v*a), SIMD3(u*h, a, v*a), SIMD3(u*a, a, v*h), SIMD3(u*a, -a, v*h)], SIMD3(u, 0, v))
            // edge ∥ X (between ±Y and ±Z faces)
            emit([SIMD3(-a, u*h, v*a), SIMD3(a, u*h, v*a), SIMD3(a, u*a, v*h), SIMD3(-a, u*a, v*h)], SIMD3(0, u, v))
        }}
        // 8 corner facets.
        for sx in signs { for sy in signs { for sz in signs {
            emit([SIMD3(sx*h, sy*a, sz*a), SIMD3(sx*a, sy*h, sz*a), SIMD3(sx*a, sy*a, sz*h)], SIMD3(sx, sy, sz))
        }}}
        return CoinMesh.Mesh(positions: pos, normals: nrm, uvs: uv, indices: idx)
    }
}
