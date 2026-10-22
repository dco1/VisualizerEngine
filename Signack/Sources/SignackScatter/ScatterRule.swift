import simd

/// One species of a planting mix — what the RULES need to know about it. The geometry is the host's;
/// `id` is the host's own handle (Daydream: an index into its species catalogue).
public struct ScatterSpecies: Sendable, Equatable {
    public var id: Int
    /// Relative abundance in the mix (any positive scale).
    public var weight: Double
    /// Canopy radius (m) of a plant at scale 1 — what spacing and exclusion measure.
    public var canopyRadius: Double
    /// Overall size of an individual: young (age 0) → mature (age 1).
    public var scaleRange: ClosedRange<Double>
    /// Baked shape variants to pick among (≥ 1).
    public var variantCount: Int
    /// Largest lean off vertical (degrees) — a wind-shaped stand, a shrub on a bank.
    public var maxTiltDegrees: Double
    /// How strongly this species gathers in drifts (0 = scattered evenly through the mix,
    /// 1 = mostly in its own clumps).
    public var clumpAffinity: Double

    public init(id: Int, weight: Double = 1, canopyRadius: Double, scaleRange: ClosedRange<Double> = 0.8...1.2,
                variantCount: Int = 1, maxTiltDegrees: Double = 4, clumpAffinity: Double = 0.5) {
        self.id = id
        self.weight = max(0, weight)
        self.canopyRadius = max(0.01, canopyRadius)
        self.scaleRange = scaleRange
        self.variantCount = max(1, variantCount)
        self.maxTiltDegrees = max(0, maxTiltDegrees)
        self.clumpAffinity = min(1, max(0, clumpAffinity))
    }
}

/// How plants gather. Clump CENTRES sit on their own world-anchored jittered lattice
/// (`spacing` apart); density is concentrated toward them with a Gaussian falloff of `radius`,
/// each clump has a dominant species, and plants nearer a clump's heart are older (bigger).
public struct ScatterClumping: Sendable, Equatable {
    /// Metres between clump centres; 0 = no clumping.
    public var spacing: Double
    /// Gaussian radius of a clump (m).
    public var radius: Double
    /// 0…1: how much of the density the clumps pull in (0 = uniform, 1 = only in clumps).
    public var strength: Double

    public init(spacing: Double = 0, radius: Double = 1.5, strength: Double = 0.6) {
        self.spacing = max(0, spacing); self.radius = max(0.1, radius); self.strength = min(1, max(0, strength))
    }

    public static let none = ScatterClumping()
    public var isOn: Bool { spacing > 0 && strength > 0 }
}

/// A planting RULESET — the whole description of what grows where. Same rule + same mask + same
/// terrain ⇒ the same points, on every run and every machine.
public struct ScatterRule: Sendable, Equatable {
    /// Plan polygons to plant (metres, x east, y north). Overlaps count once.
    public var regions: [[SIMD2<Double>]]
    /// Target plants per m² before spacing and clumping (the mean over the region).
    public var density: Double
    /// Extra metres between canopies (0 = canopies may touch). Spacing between two plants is
    /// (rᵢ + rⱼ)·`canopyOverlap` + `minSpacing`, rᵢ = canopy radius × scale.
    public var minSpacing: Double
    /// 1 = canopies just touch; < 1 lets them interlock (a hedge-like border), > 1 keeps air.
    public var canopyOverlap: Double
    /// Keep canopies this far inside the region outline (m; 0 = stems may stand on the edge).
    public var edgeMargin: Double
    public var clumping: ScatterClumping
    public var species: [ScatterSpecies]
    /// 0…1: how patchy the species mix is across the region (value-noise drifts).
    public var speciesDrift: Double
    /// Steepest ground (degrees) a plant takes; nil = any.
    public var maxSlopeDegrees: Double?
    /// Ground elevation band (m); nil = any.
    public var altitudeRange: ClosedRange<Double>?
    /// The planting's seed (Daydream: `id.seed64`, never `hashValue`).
    public var seed: UInt64

    public init(regions: [[SIMD2<Double>]], density: Double, minSpacing: Double = 0.1,
                canopyOverlap: Double = 0.85, edgeMargin: Double = 0, clumping: ScatterClumping = .none,
                species: [ScatterSpecies], speciesDrift: Double = 0.5, maxSlopeDegrees: Double? = nil,
                altitudeRange: ClosedRange<Double>? = nil, seed: UInt64) {
        self.regions = regions
        self.density = max(0, density)
        self.minSpacing = max(0, minSpacing)
        self.canopyOverlap = max(0.1, canopyOverlap)
        self.edgeMargin = max(0, edgeMargin)
        self.clumping = clumping
        self.species = species
        self.speciesDrift = min(1, max(0, speciesDrift))
        self.maxSlopeDegrees = maxSlopeDegrees
        self.altitudeRange = altitudeRange
        self.seed = seed
    }
}

/// The ground a planting stands on: height (m) at a plan point. Slope is read from it by finite
/// differences, so a host supplies one function and nothing else. Flat ground is `.flat`.
public struct ScatterTerrain: Sendable {
    public let height: @Sendable (SIMD2<Double>) -> Double
    public init(height: @escaping @Sendable (SIMD2<Double>) -> Double) { self.height = height }

    public static let flat = ScatterTerrain { _ in 0 }

    /// Slope in degrees at `p` (central differences over ±`h` metres).
    public func slopeDegrees(at p: SIMD2<Double>, h: Double = 0.25) -> Double {
        let dx = (height(p + SIMD2(h, 0)) - height(p - SIMD2(h, 0))) / (2 * h)
        let dy = (height(p + SIMD2(0, h)) - height(p - SIMD2(0, h))) / (2 * h)
        return atan((dx * dx + dy * dy).squareRoot()) * 180 / .pi
    }
}
