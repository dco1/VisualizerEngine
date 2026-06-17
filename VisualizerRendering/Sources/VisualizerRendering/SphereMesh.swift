import Foundation
import simd

// ── SphereMesh ────────────────────────────────────────────────────────────────
//
// A UV sphere in the instanced-renderer mesh format (`CoinMesh.Mesh`), authored at
// TRUE radius (the sphere/marble `CoinTransform` is a pure rotation, so the mesh
// carries the size — like `CoinMesh`). Smooth shading (vertex normal = unit
// position), wound CCW-from-outside to match `IlluminatoramaMesh.unitSphere`.

public enum SphereMesh {

    /// A sphere of the given radius. `meridians`×`parallels` = 32×16 gives a smooth
    /// specular highlight without being expensive per instance.
    public static func make(radius R: Float,
                            meridians: Int = 32,
                            parallels: Int = 16) -> CoinMesh.Mesh {
        let mer = max(3, meridians), par = max(2, parallels)
        var pos: [SIMD3<Float>] = [], nrm: [SIMD3<Float>] = []
        var uv:  [SIMD2<Float>] = [], idx: [Int32] = []
        pos.reserveCapacity((par + 1) * (mer + 1))
        for j in 0...par {
            let v = Float(j) / Float(par)
            let phi = v * .pi
            for i in 0...mer {
                let u = Float(i) / Float(mer)
                let theta = u * .pi * 2
                let n = SIMD3<Float>(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta))
                pos.append(R * n); nrm.append(n); uv.append(SIMD2(u, v))
            }
        }
        let stride = Int32(mer + 1)
        for j in 0..<par {
            for i in 0..<mer {
                let a = Int32(j) * stride + Int32(i)
                let b = Int32(j) * stride + Int32(i) + 1
                let c = (Int32(j) + 1) * stride + Int32(i)
                let d = (Int32(j) + 1) * stride + Int32(i) + 1
                idx += [a, b, c, b, d, c]            // CCW from outside
            }
        }
        return CoinMesh.Mesh(positions: pos, normals: nrm, uvs: uv, indices: idx)
    }
}
