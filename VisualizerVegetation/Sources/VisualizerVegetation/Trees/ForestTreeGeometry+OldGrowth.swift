import Foundation
import simd
import VisualizerMaterials

// OLD-GROWTH CONIFER — the conifer factory (`growConiferSkeleton` + `emitConiferWood/Sprigs`) at forest
// scale. The Christmas fir is a 1.4 m cut tree; an old-growth spruce/fir is a different animal: a long
// BARE BOLE under the first living whorl (the lower branches shaded out and died), a narrow, deep
// crown, a thick buttressed trunk, needles so dark they read as black at night. Same factory, a
// different `ConiferLook` and a different budget:
//
//  * `detail` 1 is a hero tree (a few dozen of these stand within sight of the camera); `detail` 0 is
//    a far-ring tree — far fewer whorls and sprigs, longer sprigs to hold the silhouette — so a few
//    hundred of them cost about what the hero ring does. Triangles are reported so a host can budget.
//  * Snow: boughs carry it on their UPPER faces. The per-vertex colour is frosted toward snow white
//    by how much each sprig card faces up (`frost`), so the crown reads as snow-laden without a
//    second mesh or a texture.
extension ForestTreeGeometry {

    public struct OldGrowthTree: Sendable {
        public var wood: Mesh3
        public var woodColors: [Vec3]
        public var foliage: Mesh3
        public var foliageColors: [Vec3]
        public var height: Float
        public var triangleCount: Int { (wood.indices.count + foliage.indices.count) / 3 }
    }

    /// A conifer `height` metres tall, local origin at the trunk base (+Y up). Deterministic per seed.
    public static func oldGrowthConifer(height H: Float, seed: UInt64, detail: Float = 1,
                                        frost: Float = 0.55) -> OldGrowthTree {
        let d = max(0, min(1, detail))
        var look = Self.look(for: .fir)
        var c = look.conifer ?? ForestTreeGeometry.look(for: .fir).conifer!
        // Dark needles (night), dark bark; this is the SAME dye the fir uses, taken down and cooled.
        c.needleShade = srgbToLinear(SIMD3(0.012, 0.045, 0.040))
        c.needleLit   = srgbToLinear(SIMD3(0.060, 0.170, 0.130))
        look.bark = srgbToLinear(SIMD3(0.20, 0.15, 0.12))
        // Whorl / sprig spacing are ratios of the canopy height; detail 1 ≈ 0.075 / 0.050, detail 0
        // ≈ 0.13 / 0.11 with longer sprigs so the thin crown still reads as a mass from far off.
        c.whorlSpacing = 0.13 - 0.055 * d
        c.sprigSpacing = 0.11 - 0.060 * d
        c.sprigLength  = 0.115 - 0.020 * d
        c.sprigAspect  = 0.24
        c.branchRadius = 0.0030
        c.baseRadius   = 0.20          // a narrow spire: crown radius ~0.2 × canopy height at the base
        c.convexity    = 1.15
        c.reach        = 0.97
        look.conifer = c
        look.leafCurl = 0.25

        let bottomY = H * 0.30                       // the bare bole of a shaded-out old tree
        let canopy = H - bottomY
        let sk = growConiferSkeleton(bottomY: bottomY, apexY: H, baseRadius: c.baseRadius * canopy,
                                     look: c, density: 1, seed: seed)
        var soup = Soup()
        soup.skipClearingCull = true
        emitConiferWood(&soup, world: matrix_identity_float4x4, sk: sk, look: look,
                        trunkR: 0.017 * H, trunkBaseY: 0)
        emitConiferSprigs(&soup, world: matrix_identity_float4x4, sk: sk, look: look, seed: seed)
        let parts = soup.splitMeshes()

        // Frost the foliage by how much each vertex faces up.
        var colors = parts.foliageColors
        let snow = Vec3(0.78, 0.84, 0.92)
        for i in colors.indices where i < parts.foliage.normals.count {
            let up = max(0, parts.foliage.normals[i].y)
            let f = Double(frost) * pow(up, 1.6)
            colors[i] = colors[i] * (1 - f) + snow * f
        }
        return OldGrowthTree(wood: parts.wood, woodColors: parts.woodColors,
                             foliage: parts.foliage, foliageColors: colors, height: H)
    }
}
