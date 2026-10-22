import simd

/// Who grows at a candidate point, and how that individual differs from its siblings.
///
/// Species: the rule's weights, modulated by slow value-noise DRIFTS (one field per species, so
/// the mix gathers into patches) and by the clump the point sits in (its dominant species is
/// favoured in proportion to that species' `clumpAffinity`). Individual: age (older toward a
/// clump's heart), scale from age, yaw, lean, a tint shared partly with its neighbours (a patch of
/// healthier ground) and a wind phase — all from the candidate's own seeded stream.
enum ScatterIndividual {

    /// Drift field scale: ≈ 6 m patches.
    static let driftFrequency = 0.17

    static func pickSpecies(_ rule: ScatterRule, at p: SIMD2<Double>, clump: ScatterClumps.Sample,
                            u: Double) -> Int? {
        var w = rule.species.map(\.weight)
        for i in w.indices {
            if rule.speciesDrift > 0 {
                let n = ScatterNoise.value(p * driftFrequency, seed: rule.seed &+ 0xD71F7 &+ UInt64(i) &* 977)
                w[i] *= (1 - rule.speciesDrift) + rule.speciesDrift * 2 * n
            }
            if let dom = clump.dominant {
                let a = rule.species[i].clumpAffinity * clump.weight
                w[i] *= i == dom ? 1 + 6 * a : max(0.05, 1 - 0.7 * a)
            }
        }
        return ScatterClumps.weightedIndex(w, u)
    }

    /// The individual for species index `si` at `p`, drawn from `rng` (the candidate's stream).
    static func make(_ rule: ScatterRule, species si: Int, at p: SIMD2<Double>, ground: Double,
                     clump: ScatterClumps.Sample, key: UInt64, rng: inout ScatterRandom) -> ScatterPoint {
        let sp = rule.species[si]
        let variant = Int(rng.next() % UInt64(sp.variantCount))
        // Older toward a clump's heart; elsewhere mostly young-to-middle with a tail of old ones.
        let u = rng.unit()
        let age = min(1, max(0, 0.35 * clump.weight + (1 - 0.35 * clump.weight) * u.squareRoot() * 0.95 + 0.05 * rng.unit()))
        let scale = sp.scaleRange.lowerBound + (sp.scaleRange.upperBound - sp.scaleRange.lowerBound) * age
        let yaw = rng.unit() * 2 * .pi
        let t = rng.unit()
        let tilt = sp.maxTiltDegrees * .pi / 180 * t * t
        let azimuth = rng.unit() * 2 * .pi
        // Tint: a neighbourhood term (patches of richer / thirstier ground) + the plant's own.
        let patch = ScatterNoise.value(p * 0.31, seed: rule.seed &+ 0x7147) - 0.5
        let value = 1 + 0.10 * patch + 0.12 * (rng.unit() - 0.5)
        let warm = 0.08 * (rng.unit() - 0.5) - 0.05 * patch
        let tint = SIMD3<Float>(Float(value * (1 + warm)), Float(value), Float(value * (1 - warm)))
        let wind = Float(rng.unit() * 2 * .pi)
        return ScatterPoint(position: p, groundHeight: ground, species: sp.id, variant: variant,
                            age: age, scale: scale, radius: sp.canopyRadius * scale,
                            yaw: yaw, tilt: tilt, tiltAzimuth: azimuth, tint: tint,
                            windPhase: wind, key: key)
    }
}
