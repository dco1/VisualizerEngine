import Foundation
import simd

// Species identity, look profiles, sRGB→linear look-up.
extension ForestTreeGeometry {
    // ===== Species+TreeLook+look+TreeSite (lines 1057-1169) =====
    public enum Species: CaseIterable, Sendable { case oak, birch, maple, orange, lemon, elderberry, fir
        // The street trees of a warm city (DH-1649): a broad dark evergreen (fig, camphor, magnolia), the
        // ferny jacaranda, the columnar Italian cypress, and the palms and palm-likes
        // (`ForestTreeGeometry+Palm`).
        case fig, jacaranda, cypress, fanPalm, kingPalm, datePalm, yucca
    }

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
        /// Present for a CONIFER: it grows by whorls of branches carrying needled sprigs
        /// (`ForestTreeGeometry+Conifer`), not by the broadleaf crown-arm + volumetric leaf fill
        /// that every other field above describes. `nil` for every broadleaf — byte-identical.
        public var conifer: ConiferLook? = nil
        /// Present for a PALM (or a palm-like rosette plant): one unbranched stem and a single crown of
        /// fronds (`ForestTreeGeometry+Palm`). `nil` for everything else.
        public var palm: PalmLook? = nil
    }

    /// How a conifer is grown. Lengths are FRACTIONS of the canopy height, so a desk fir and a hall
    /// fir are the same tree at two sizes. `needleShade`/`needleLit` are linear albedo for a sprig's
    /// base and tip.
    public struct ConiferLook: Sendable {
        public var whorlSpacing: Float
        public var sprigSpacing: Float
        public var sprigLength: Float
        public var sprigAspect: Float
        public var branchRadius: Float
        public var baseRadius: Float        // canopy radius at its base, per canopy height
        public var convexity: Float
        public var reach: Float
        public var needleShade: SIMD3<Float>
        public var needleLit: SIMD3<Float>
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
                bark: srgbToLinear(SIMD3(0.40, 0.33, 0.26)), isBirch: false,   // DH-0629: was (0.32,0.28,0.22) — under the vertex seam + shader furrow darkening the bole rendered near-black
                // An open-grown oak SPREADS — broader than it is tall (DH-0629).
                species: .oak, crownWidthMult: 1.12)
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
                species: .maple, crownWidthMult: 1.06)
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
        case .fir:
            // A cut fir as sold for Christmas — dense, deep blue-green, tiered by its whorls.
            return TreeLook(
                trunkHeightFrac: 0.10, flareBase: 1.20, trunkTopMult: 0.10,
                maxDepth: 1, primaryCountMin: 5, primaryCountMax: 8,
                primaryPitchMin: 0, primaryPitchRange: 0,
                secondaryPitchMin: 0, secondaryPitchRange: 0,
                lobePitchBoost: 0, leafCardSize: 0.08, leafDensityMult: 1,
                leafAspectW: 0.3, leafAspectH: 1, opposite: false,
                internodeFrac: 0.3, petioleFrac: 0.1,
                leaf: srgbToLinear(SIMD3(0.075, 0.30, 0.17)),
                bark: srgbToLinear(SIMD3(0.17, 0.11, 0.08)), isBirch: false,
                species: .fir, leafCurl: 0.20,
                conifer: ConiferLook(whorlSpacing: 0.040, sprigSpacing: 0.006, sprigLength: 0.074,
                                     sprigAspect: 0.21, branchRadius: 0.0032, baseRadius: 0.44,
                                     convexity: 0.95, reach: 0.95,
                                     needleShade: srgbToLinear(SIMD3(0.020, 0.100, 0.075)),
                                     needleLit: srgbToLinear(SIMD3(0.110, 0.330, 0.200))))
        case .fig:
            // Indian laurel fig / camphor / southern magnolia: the broad, dense, DARK evergreen dome
            // of a warm-city street. An oak's habit with a heavier, glossier, smaller-leaved crown.
            var l = look(for: .oak)
            l.species = .fig
            l.trunkHeightFrac = 0.30; l.primaryCountMin = 6; l.primaryCountMax = 8
            l.leafCardSize = 0.090; l.leafDensityMult = 6.5
            l.leafAspectW = 0.80; l.leafAspectH = 1.30; l.leafCurl = 0.20
            l.leaf = srgbToLinear(SIMD3(0.085, 0.24, 0.085))
            l.bark = srgbToLinear(SIMD3(0.52, 0.49, 0.44))      // smooth pale grey
            l.crownWidthMult = 1.18
            return l
        case .jacaranda:
            // Open, spreading and airy: long bare limbs, the foliage a fine light-green lace of
            // bipinnate leaves (many small narrow blades), the crown wider than tall.
            var l = look(for: .elderberry)
            l.species = .jacaranda
            l.fruit = nil
            l.trunkHeightFrac = 0.34; l.primaryCountMin = 4; l.primaryCountMax = 6
            l.primaryPitchMin = 0.50; l.primaryPitchRange = 0.45
            l.leafCardSize = 0.085; l.leafDensityMult = 5.6
            l.leafAspectW = 0.42; l.leafAspectH = 1.70; l.leafCurl = 0.30
            l.leaf = srgbToLinear(SIMD3(0.20, 0.42, 0.14))
            l.bark = srgbToLinear(SIMD3(0.40, 0.37, 0.33))
            l.crownWidthMult = 1.30
            return l
        case .cypress:
            // Italian cypress: a dark column a dozen metres tall and barely two wide, clothed to
            // near the ground. A conifer whose envelope holds its width almost to the tip.
            var l = look(for: .fir)
            l.species = .cypress
            l.leaf = srgbToLinear(SIMD3(0.060, 0.17, 0.085))
            l.bark = srgbToLinear(SIMD3(0.30, 0.25, 0.20))
            l.conifer = ConiferLook(whorlSpacing: 0.030, sprigSpacing: 0.011, sprigLength: 0.034,
                                    sprigAspect: 0.50, branchRadius: 0.0022, baseRadius: 0.085,
                                    convexity: 0.38, reach: 0.985,
                                    needleShade: srgbToLinear(SIMD3(0.022, 0.075, 0.040)),
                                    needleLit: srgbToLinear(SIMD3(0.085, 0.215, 0.105)))
            return l
        case .fanPalm:
            // Mexican fan palm — the skyline palm: a very tall, very slender stem under a small
            // round head of fans, with a short thatch of dead ones beneath it.
            return palmLook(.fanPalm, leaf: SIMD3(0.12, 0.23, 0.10), bark: SIMD3(0.40, 0.34, 0.28),
                            PalmLook(frond: .fan, referenceHeight: 20, trunkRadius: 0.17, baseFlare: 1.7,
                                     frondCount: 30, frondLength: 2.3, droop: 0.45, lowestElevation: -0.75,
                                     skirt: 0.5, skirtLength: 1.1,
                                     leafTip: srgbToLinear(SIMD3(0.21, 0.31, 0.14)),
                                     dead: srgbToLinear(SIMD3(0.40, 0.32, 0.21))))
        case .kingPalm:
            // King / queen palm: a smooth ringed grey stem, a green crownshaft, and a loose crown
            // of long arching feathers.
            return palmLook(.kingPalm, leaf: SIMD3(0.09, 0.21, 0.08), bark: SIMD3(0.50, 0.48, 0.44),
                            PalmLook(frond: .feather, referenceHeight: 11, trunkRadius: 0.14, baseFlare: 1.6,
                                     frondCount: 15, frondLength: 3.0, droop: 0.55, lowestElevation: -0.45,
                                     crownshaft: 1.0,
                                     leafTip: srgbToLinear(SIMD3(0.17, 0.30, 0.12)),
                                     dead: srgbToLinear(SIMD3(0.50, 0.42, 0.24)),
                                     shaft: srgbToLinear(SIMD3(0.30, 0.42, 0.20))))
        case .datePalm:
            // Canary Island date palm: a stout, rough stem and an enormous dense sphere of feathers.
            return palmLook(.datePalm, leaf: SIMD3(0.08, 0.19, 0.08), bark: SIMD3(0.36, 0.30, 0.24),
                            PalmLook(frond: .feather, referenceHeight: 13, trunkRadius: 0.42, baseFlare: 1.25,
                                     frondCount: 54, frondLength: 4.4, droop: 0.38, lowestElevation: -0.70,
                                     leafTip: srgbToLinear(SIMD3(0.15, 0.27, 0.11)),
                                     dead: srgbToLinear(SIMD3(0.48, 0.40, 0.22))))
        case .yucca:
            // Spanish dagger / giant yucca: a thick stem forking into a few heads, each a rosette of
            // stiff grey-green swords with the dead ones folded down beneath.
            return palmLook(.yucca, leaf: SIMD3(0.15, 0.24, 0.15), bark: SIMD3(0.42, 0.37, 0.31),
                            PalmLook(frond: .sword, referenceHeight: 4.5, trunkRadius: 0.11, baseFlare: 1.8,
                                     frondCount: 84, frondLength: 0.75, droop: 0.5, lowestElevation: -1.0,
                                     heads: 4,
                                     leafTip: srgbToLinear(SIMD3(0.26, 0.34, 0.22)),
                                     dead: srgbToLinear(SIMD3(0.55, 0.47, 0.30))))
        }
    }

    /// A palm's `TreeLook`: only the colours, species and `palm` are read — the broadleaf fields are
    /// the oak's, unused.
    private static func palmLook(_ sp: Species, leaf: SIMD3<Float>, bark: SIMD3<Float>, _ palm: PalmLook) -> TreeLook {
        var l = look(for: .oak)
        l.species = sp
        l.leaf = srgbToLinear(leaf); l.bark = srgbToLinear(bark)
        l.crownWidthMult = 1
        l.palm = palm
        return l
    }

}
