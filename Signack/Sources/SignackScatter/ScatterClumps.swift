import Foundation
import simd

/// The clump field: where a planting gathers, which species each drift belongs to, and how far a
/// plant is from a clump's heart. Clump centres live on their OWN world-anchored jittered lattice
/// (one candidate per `spacing` cell, ~80 % present, radius jittered ±35 %), so editing a
/// region's edge never moves a clump in its middle.
struct ScatterClumps {
    let rule: ScatterClumping
    let seed: UInt64
    let speciesWeights: [Double]
    /// Mean of the raw falloff over the plane — divides it out so clumping MOVES density
    /// rather than adding or removing it.
    let meanFalloff: Double

    static let presence = 0.8

    init(_ rule: ScatterClumping, species: [ScatterSpecies], seed: UInt64) {
        self.rule = rule
        self.seed = seed &+ 0xC1A0_0000
        speciesWeights = species.map(\.weight)
        let area = 2 * Double.pi * rule.radius * rule.radius
        meanFalloff = rule.isOn ? min(1, Self.presence * area / max(1e-6, rule.spacing * rule.spacing)) : 1
    }

    struct Sample {
        /// Gaussian falloff of the nearest clump (0 far away … 1 at its heart).
        var weight: Double
        /// Index into the rule's species of the clump's dominant species (nil = no clump near).
        var dominant: Int?
        /// The density multiplier at this point (mean ≈ 1 over the plane).
        var densityFactor: Double
    }

    func sample(_ p: SIMD2<Double>) -> Sample {
        guard rule.isOn else { return Sample(weight: 0, dominant: nil, densityFactor: 1) }
        let s = rule.spacing
        let cx = Int((p.x / s).rounded(.down)), cy = Int((p.y / s).rounded(.down))
        var best = 0.0, dominant: Int?
        for iy in (cy - 1)...(cy + 1) {
            for ix in (cx - 1)...(cx + 1) {
                var r = ScatterRandom(seed: seed, cell: ix, iy)
                let c = SIMD2((Double(ix) + 0.1 + 0.8 * r.unit()) * s, (Double(iy) + 0.1 + 0.8 * r.unit()) * s)
                let present = r.unit() < Self.presence
                let radius = rule.radius * (0.65 + 0.7 * r.unit())
                let pick = r.unit()
                guard present else { continue }
                let d2 = simd_distance_squared(p, c)
                let w = exp(-d2 / (2 * radius * radius))
                if w > best { best = w; dominant = Self.weightedIndex(speciesWeights, pick) }
            }
        }
        let factor = (1 - rule.strength) + rule.strength * best / meanFalloff
        return Sample(weight: best, dominant: dominant, densityFactor: factor)
    }

    /// The index `u ∈ [0,1)` lands on in a weight table (nil when every weight is 0).
    static func weightedIndex(_ weights: [Double], _ u: Double) -> Int? {
        let total = weights.reduce(0, +)
        guard total > 0 else { return nil }
        var x = u * total
        for (i, w) in weights.enumerated() {
            if x < w { return i }
            x -= w
        }
        return weights.lastIndex(where: { $0 > 0 })
    }
}
