import Foundation
import simd
import VisualizerMaterials

extension ForestTreeGeometry.Soup {
    /// A baked tree as `Mesh3`s for a host that draws `Mesh3` (the potted plants): the FOLIAGE
    /// triangles (the soup's `triLeaf` flag — `foliageMark > 0.5` at emission) with one albedo per
    /// vertex, and everything else (trunk, branches) as WOOD. Split by the LEAF flag, not `triWood`:
    /// that one is only set under the tree-wind context a yard tree bakes in, which a potted bake
    /// does not have. Normals are the soup's smooth shading normals, not re-derived — they
    /// are what the tree was authored to shade with.
    public func splitMeshes() -> (wood: Mesh3, woodColors: [Vec3], foliage: Mesh3, foliageColors: [Vec3]) {
        var wood = Mesh3(), foliage = Mesh3()
        var colors: [Vec3] = [], woodColors: [Vec3] = []
        func v(_ p: SIMD3<Float>) -> Vec3 { Vec3(Double(p.x), Double(p.y), Double(p.z)) }
        let tris = indices.count / 3
        for t in 0 ..< tris {
            let isWood = !(t < triLeaf.count ? triLeaf[t] : false)
            for k in 0 ..< 3 {
                let i = Int(indices[3 * t + k])
                if isWood {
                    wood.indices.append(UInt32(wood.positions.count))
                    wood.positions.append(v(positions[i])); wood.normals.append(v(normals[i]))
                    wood.uvs.append(Vec2(0, 0))
                    woodColors.append(v(self.colors[i]))
                } else {
                    foliage.indices.append(UInt32(foliage.positions.count))
                    foliage.positions.append(v(positions[i])); foliage.normals.append(v(normals[i]))
                    foliage.uvs.append(Vec2(0, 0))
                    colors.append(v(self.colors[i]))
                }
            }
        }
        return (wood, woodColors, foliage, colors)
    }
}
