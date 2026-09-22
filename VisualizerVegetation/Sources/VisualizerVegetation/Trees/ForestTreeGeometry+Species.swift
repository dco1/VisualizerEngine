import Foundation
import simd

// Species identity, look profiles, sRGB→linear look-up.
extension ForestTreeGeometry {
    // ===== Species+TreeLook+look+TreeSite (lines 1057-1169) =====
    public enum Species: CaseIterable, Sendable { case oak, birch, maple, orange, lemon, elderberry }

    /// Ported subset of `TreeGeometry.LookProfile` — proportions + species
    /// identity. Colours are converted sRGB→linear for the Illuminatorama
    /// linear pipeline.
    public struct TreeLook {
        public var trunkHeightFrac: Float
        public var flareBase: Float
        public var trunkTopMult: Float
        public var maxDepth: Int
        public var primaryCountMin: Int
        public var primaryCountMax: Int
        public var primaryPitchMin: Float
        public var primaryPitchRange: Float
        public var secondaryPitchMin: Float
        public var secondaryPitchRange: Float
        public var lobePitchBoost: Float
        public var leafCardSize: Float
        public var leafDensityMult: Float
        public var leafAspectW: Float
        public var leafAspectH: Float
        public var opposite: Bool           // phyllotaxis: opposite (maple) vs alternate
        public var internodeFrac: Float
        public var petioleFrac: Float
        public var leaf: SIMD3<Float>       // linear leaf albedo
        public var bark: SIMD3<Float>       // linear bark albedo
        public var isBirch: Bool
        public var species: Species         // blade-outline selector (oak/birch/maple/citrus)
        /// Longitudinal gravity-droop of every leaf blade — the DH-0488 technique brought to trees
        /// (DH-0570): the midrib bows out of the flat plane along −bentNormal accumulating to the
        /// tip as `v²` (a self-weighted cantilever), so a near leaf reads as a solid 3D arch
        /// instead of a flat facet. Zero extra triangles; `0` reproduces the old flat blade.
        public var leafCurl: Float = 0.22
        /// Ripe fruit hung in the crown — nil for the broadleaves. `count` fruits of `radius` m
        /// (scaled with the tree), each an ellipsoid squashed/stretched by `elongate` (1 = round
        /// orange, >1 = drawn-out lemon), coloured `color` (linear). The fruit is what tells an
        /// orange from a lemon and either from an oak.
        public var fruit: FruitLook? = nil
        /// Crown width against the height-derived default. 1 for every tree-form species; a
        /// shrub-tree (elderberry) is wider than its height suggests.
        public var crownWidthMult: Float = 1
    }

    /// Descriptor for a citrus tree's crown fruit (see `TreeLook.fruit`).
    public struct FruitLook {
        public var color: SIMD3<Float>   // linear fruit albedo
        public var radius: Float         // fruit radius in metres (for a ~6.5 m tree)
        public var elongate: Float       // vertical stretch: 1 = round (orange), >1 = ovoid (lemon)
        public var count: Int            // how many fruit to scatter through the crown
        /// Where on the crown they sit. Fruit HANGS in the outer-lower crown; a flower cluster is
        /// CROWNING — borne on the outside of the upper crown, facing the sky.
        public var placement: Placement = .hanging
        public enum Placement { case hanging, crowning }
    }

    public static func look(for sp: Species) -> TreeLook {
        switch sp {
        case .oak:
            return TreeLook(
                trunkHeightFrac: 0.38, flareBase: 1.45, trunkTopMult: 0.85,
                maxDepth: 3, primaryCountMin: 5, primaryCountMax: 7,
                primaryPitchMin: 0.80, primaryPitchRange: 0.30,
                secondaryPitchMin: 0.50, secondaryPitchRange: 0.45,
                lobePitchBoost: 0.22, leafCardSize: 0.100, leafDensityMult: 4.00,
                leafAspectW: 1.00, leafAspectH: 1.30, opposite: false,
                internodeFrac: 0.38, petioleFrac: 0.22,
                // LEAF round-7: side-view canopy still bleached to pale khaki and
                // the residual was ALBEDO (the SSS roll-off landed; an exposure
                // pull didn't move it). Drop value ~15% and bias toward green
                // chroma (R/B down harder than G) so the sunlit mass finishes as
                // saturated golden-green, not cream.
                leaf: srgbToLinear(SIMD3(0.145, 0.34, 0.11)),
                bark: srgbToLinear(SIMD3(0.32, 0.28, 0.22)), isBirch: false,
                species: .oak)
        case .birch:
            return TreeLook(
                trunkHeightFrac: 0.60, flareBase: 1.18, trunkTopMult: 0.78,
                maxDepth: 4, primaryCountMin: 4, primaryCountMax: 6,
                primaryPitchMin: 0.10, primaryPitchRange: 0.40,
                secondaryPitchMin: 0.45, secondaryPitchRange: 0.45,
                lobePitchBoost: 0.0, leafCardSize: 0.062, leafDensityMult: 5.50,
                leafAspectW: 0.95, leafAspectH: 1.20, opposite: false,
                internodeFrac: 0.38, petioleFrac: 0.30,
                // LEAF: darkened from (0.36,0.52,0.22) — vivid spring-green read
                // as neon yellow-green in linear pipeline. Real birch mid-leaf at
                // golden hour: muted mid-green, sRGB ≈ (0.30,0.42,0.18). ✓
                // BARK: round-3 retune. The old (0.72,0.70,0.66) read as a
                // pinkish-TAN trunk in the hero/side crops — R>G>B is warm, and at
                // 3–10 m the dark lenticels averaged out to a flesh tone, not white
                // birch. Real paper birch is a NEUTRAL-COOL high-value white: lift
                // the base AND flip the bias so B≈G≥R (faint cool cast), so even
                // after the shader's per-pixel paper tint + lenticel darkening the
                // trunk reads unmistakably WHITE at distance. sRGB ≈ (0.82,0.83,0.84).
                // LEAF round-7: −15% value, +green chroma (see oak note).
                leaf: srgbToLinear(SIMD3(0.24, 0.40, 0.14)),
                bark: srgbToLinear(SIMD3(0.82, 0.83, 0.84)), isBirch: true,
                species: .birch)
        case .maple:
            return TreeLook(
                trunkHeightFrac: 0.42, flareBase: 1.38, trunkTopMult: 0.82,
                maxDepth: 3, primaryCountMin: 4, primaryCountMax: 6,
                primaryPitchMin: 0.45, primaryPitchRange: 0.45,
                secondaryPitchMin: 0.35, secondaryPitchRange: 0.40,
                lobePitchBoost: 0.0, leafCardSize: 0.115, leafDensityMult: 3.50,
                leafAspectW: 1.10, leafAspectH: 1.00, opposite: true,
                internodeFrac: 0.38, petioleFrac: 0.26,
                // LEAF: darkened from (0.28,0.46,0.18) — still too vivid.
                // Real big-leaf maple foliage at golden hour: muted olive green,
                // sRGB ≈ (0.24,0.40,0.15). ✓
                // LEAF round-7: −15% value, +green chroma (see oak note).
                leaf: srgbToLinear(SIMD3(0.19, 0.38, 0.12)),
                bark: srgbToLinear(SIMD3(0.42, 0.40, 0.36)), isBirch: false,
                species: .maple)
        case .orange:
            return TreeLook(
                // Citrus habit: a SHORT clear bole, low-branching into a DENSE, deep, rounded
                // evergreen crown of many SMALL glossy blades. leafCardSize small + high density
                // + short internodes read as the packed citrus canopy, not an open broadleaf.
                trunkHeightFrac: 0.32, flareBase: 1.30, trunkTopMult: 0.86,
                maxDepth: 3, primaryCountMin: 5, primaryCountMax: 7,
                primaryPitchMin: 0.55, primaryPitchRange: 0.35,
                secondaryPitchMin: 0.45, secondaryPitchRange: 0.40,
                lobePitchBoost: 0.0, leafCardSize: 0.055, leafDensityMult: 5.50,
                leafAspectW: 0.78, leafAspectH: 1.35, opposite: false,
                internodeFrac: 0.30, petioleFrac: 0.20,
                // Deep glossy citrus green (darker + less chroma than a broadleaf's spring leaf).
                leaf: srgbToLinear(SIMD3(0.11, 0.26, 0.09)),
                // Smooth grey-brown citrus bark.
                bark: srgbToLinear(SIMD3(0.36, 0.32, 0.27)), isBirch: false,
                species: .orange, leafCurl: 0.26,
                fruit: FruitLook(color: srgbToLinear(SIMD3(0.95, 0.42, 0.05)),
                                 radius: 0.046, elongate: 1.0, count: 16))
        case .lemon:
            return TreeLook(
                trunkHeightFrac: 0.32, flareBase: 1.28, trunkTopMult: 0.86,
                maxDepth: 3, primaryCountMin: 5, primaryCountMax: 7,
                primaryPitchMin: 0.55, primaryPitchRange: 0.35,
                secondaryPitchMin: 0.45, secondaryPitchRange: 0.40,
                lobePitchBoost: 0.0, leafCardSize: 0.052, leafDensityMult: 5.30,
                leafAspectW: 0.76, leafAspectH: 1.38, opposite: false,
                internodeFrac: 0.30, petioleFrac: 0.20,
                // A touch lighter / yellower-green than the orange's darker crown.
                leaf: srgbToLinear(SIMD3(0.16, 0.30, 0.10)),
                bark: srgbToLinear(SIMD3(0.40, 0.36, 0.30)), isBirch: false,
                species: .lemon, leafCurl: 0.26,
                // Lemon fruit: paler yellow, drawn out into an ovoid (elongate > 1).
                fruit: FruitLook(color: srgbToLinear(SIMD3(0.92, 0.80, 0.12)),
                                 radius: 0.041, elongate: 1.38, count: 14))
        case .elderberry:
            return TreeLook(
                // Habit: hardly any bole — several stems break almost at the ground into a wide,
                // loose, rounded crown. Foliage is pinnate: many narrow lanceolate leaflets, a
                // lighter yellow-green than the citrus beside it.
                trunkHeightFrac: 0.20, flareBase: 1.25, trunkTopMult: 0.88,
                maxDepth: 3, primaryCountMin: 6, primaryCountMax: 8,
                primaryPitchMin: 0.60, primaryPitchRange: 0.40,
                secondaryPitchMin: 0.45, secondaryPitchRange: 0.45,
                lobePitchBoost: 0.10, leafCardSize: 0.115, leafDensityMult: 6.00,
                leafAspectW: 0.58, leafAspectH: 1.55, opposite: true,
                internodeFrac: 0.32, petioleFrac: 0.22,
                leaf: srgbToLinear(SIMD3(0.22, 0.40, 0.12)),
                // Grey-brown, corky, deeply furrowed.
                bark: srgbToLinear(SIMD3(0.40, 0.36, 0.30)), isBirch: false,
                species: .elderberry, leafCurl: 0.24,
                // The elderflower: flat-topped cream-white cymes a hand-span across, carried on
                // the OUTSIDE of the upper crown. The crown-fruit body squashed flat is that shape.
                // The cream is stated in LINEAR 8-bit steps on purpose (217, 201, 146 over 255 ≈ sRGB
                // 0.93/0.90/0.78): the leaf group stores colour DIVIDED by the leaf albedo, and a
                // channel sitting near an x.5 step flips a byte on the way back
                // (`testSplitBakeReproducesTheDeferredProductExactly` caught 24 of them).
                fruit: FruitLook(color: SIMD3(217, 201, 146) / 255,
                                 radius: 0.11, elongate: 0.34, count: 120, placement: .crowning),
                crownWidthMult: 1.35)
        }
    }

}
