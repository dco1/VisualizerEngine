import Foundation
import simd

// UVApply — island layouts and writing UVs into a Mesh3.
//
// Guarantees: `applying` / `withCornerUVs` keep every triangle, its winding and its positions, and only
// duplicate vertices where two corners that shared a vertex now disagree about its UV (seams). Normals are
// copied when the mesh had one per vertex. Doesn't: merge vertices that were already separate.

/// An island plus one UV per local vertex.
public struct UVIslandLayout: Sendable {
    public var island: UVIsland
    public var uv: [Vec2]
    public init(island: UVIsland, uv: [Vec2]) { self.island = island; self.uv = uv }

    /// (min, max) of the UVs.
    public var bounds: (min: Vec2, max: Vec2) {
        var lo = Vec2(repeating: .infinity), hi = Vec2(repeating: -.infinity)
        for p in uv { lo = simd.min(lo, p); hi = simd.max(hi, p) }
        return (lo, hi)
    }
    public var size: Vec2 { let b = bounds; return b.max - b.min }

    /// Apply `f` to every UV.
    public func mapped(_ f: (Vec2) -> Vec2) -> UVIslandLayout { UVIslandLayout(island: island, uv: uv.map(f)) }

    /// Signed UV area summed over the island's triangles (positive when winding is kept).
    public func uvArea(_ mesh: Mesh3) -> Double {
        var a = 0.0
        for k in 0..<island.triangles.count {
            let p = (0..<3).map { uv[island.corners[3 * k + $0]] }
            a += cross2(p[1] - p[0], p[2] - p[0]) / 2
        }
        return a
    }
}

public extension Mesh3 {
    /// A copy of the mesh with island layouts written as UVs; vertices are duplicated along seams.
    func applying(_ layouts: [UVIslandLayout]) -> Mesh3 {
        struct Key: Hashable { let island: Int, local: Int, vertex: UInt32 }
        var out = Mesh3()
        let keepNormals = normals.count == positions.count
        var ids: [Key: UInt32] = [:]
        var indices = [UInt32](repeating: 0, count: self.indices.count)
        for (li, lay) in layouts.enumerated() {
            for (k, t) in lay.island.triangles.enumerated() {
                for c in 0..<3 {
                    let v = self.indices[3 * t + c], local = lay.island.corners[3 * k + c]
                    let key = Key(island: li, local: local, vertex: v)
                    if let id = ids[key] { indices[3 * t + c] = id; continue }
                    let id = UInt32(out.positions.count)
                    out.positions.append(positions[Int(v)])
                    if keepNormals { out.normals.append(normals[Int(v)]) }
                    out.uvs.append(lay.uv[local])
                    ids[key] = id; indices[3 * t + c] = id
                }
            }
        }
        out.indices = indices
        return out
    }

    /// A copy with one explicit UV per triangle corner (`3 × triangleCount`), vertices split where two
    /// corners sharing a vertex disagree (equal UVs, to 1e-9, stay shared).
    func withCornerUVs(_ cornerUVs: [Vec2]) -> Mesh3 {
        precondition(cornerUVs.count == indices.count, "one UV per index")
        struct Key: Hashable { let v: UInt32, u: Int64, w: Int64 }
        var out = Mesh3()
        let keepNormals = normals.count == positions.count
        var ids: [Key: UInt32] = [:]
        for (i, v) in indices.enumerated() {
            let uv = cornerUVs[i]
            let key = Key(v: v, u: Int64((uv.x * 1e9).rounded()), w: Int64((uv.y * 1e9).rounded()))
            if let id = ids[key] { out.indices.append(id); continue }
            let id = UInt32(out.positions.count)
            out.positions.append(positions[Int(v)])
            if keepNormals { out.normals.append(normals[Int(v)]) }
            out.uvs.append(uv)
            ids[key] = id; out.indices.append(id)
        }
        return out
    }

    /// Layouts of the mesh's *existing* UVs, one per UV island (cut where the UVs break).
    func uvLayouts(tolerance: Double = 1e-6) -> [UVIslandLayout] {
        let topo = UVTopology(self)
        return topo.islands(topo.seams(uvBoundaryOf: self, tolerance: tolerance)).map { isl in
            var uv = [Vec2](repeating: .zero, count: isl.vertexCount)
            for k in 0..<isl.triangles.count {
                for c in 0..<3 where uvs.count == positions.count {
                    uv[isl.corners[3 * k + c]] = uvs[Int(indices[3 * isl.triangles[k] + c])]
                }
            }
            return UVIslandLayout(island: isl, uv: uv)
        }
    }
}
