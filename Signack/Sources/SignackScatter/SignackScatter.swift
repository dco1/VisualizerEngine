import simd

/// **Signack S6 — rule-driven scatter** (Daydream DH-1461): the job Unreal's PCG framework does
/// for a yard, as a pure function. A planting is a RULESET (`ScatterRule`); the scatter turns it
/// into individuals (`ScatterPoint`) that a renderer draws as instances of a few baked shapes.
///
///     candidates   world-anchored jittered cells (the wildflower sowing's scheme, lifted:
///                  `WildflowerPlantingMesh.plants`) at 4× the target density, each with its own
///                  seeded stream and a priority key — so editing a region's edge never reshuffles
///                  the plants in its middle
///     filters      inside a region · clear of the mask (canopy-sized) · inside the edge margin ·
///                  slope / altitude limits
///     density      kept with probability density·clumpFactor / candidateDensity (clumps MOVE
///                  density toward their hearts; the mean is the rule's)
///     species      weighted mix × drift patches × the clump's dominant species
///     spacing      blue noise by greedy elimination in priority order: a candidate is dropped when
///                  an already-accepted plant stands closer than (rᵢ + rⱼ)·overlap + minSpacing
///                  (overlap tightens by up to 35 % toward a clump's heart — a drift interlocks)
///     individual   age, scale, yaw, lean, tint, wind phase from the candidate's stream
///
/// Deterministic: every draw is a function of (seed, world cell), and acceptance runs in a fixed
/// total order, so the same rule, mask and terrain give bit-identical points on every run.
public enum SignackScatter {

    /// Candidates per target plant — headroom for the spacing and clumping filters to choose from.
    public static let oversample = 4.0
    /// A runaway guard: at most this many candidate cells are examined (≈ 1.3 ha at 4 /m² × 4).
    public static let maxCandidates = 2_000_000

    public static func scatter(_ rule: ScatterRule, mask: ScatterMask = ScatterMask(),
                               terrain: ScatterTerrain = .flat) -> [ScatterPoint] {
        let regions = rule.regions.filter { $0.count >= 3 && ScatterGeometry.area($0) > 1e-4 }
        guard rule.density > 0, !rule.species.isEmpty, rule.species.contains(where: { $0.weight > 0 }),
              let box = ScatterGeometry.bounds(regions) else { return [] }
        let candidateDensity = rule.density * oversample
        let cell = 1 / candidateDensity.squareRoot()
        let ix0 = Int((box.lo.x / cell).rounded(.down)), ix1 = Int((box.hi.x / cell).rounded(.up))
        let iy0 = Int((box.lo.y / cell).rounded(.down)), iy1 = Int((box.hi.y / cell).rounded(.up))
        guard (ix1 - ix0 + 1) * (iy1 - iy0 + 1) <= maxCandidates else { return [] }

        let clumps = ScatterClumps(rule.clumping, species: rule.species, seed: rule.seed)
        struct Candidate { var point: ScatterPoint; var priority: UInt64; var tight: Double }
        var candidates: [Candidate] = []
        for iy in iy0...iy1 {
            for ix in ix0...ix1 {
                var rng = ScatterRandom(seed: rule.seed, cell: ix, iy)
                let p = SIMD2((Double(ix) + rng.unit()) * cell, (Double(iy) + rng.unit()) * cell)
                let priority = rng.next()
                let keep = rng.unit(), pick = rng.unit()
                guard regions.contains(where: { ScatterGeometry.contains($0, p) }) else { continue }
                let clump = clumps.sample(p)
                guard keep < clump.densityFactor / oversample else { continue }
                guard let si = ScatterIndividual.pickSpecies(rule, at: p, clump: clump, u: pick) else { continue }
                let ground = terrain.height(p)
                if let band = rule.altitudeRange, !band.contains(ground) { continue }
                if let maxSlope = rule.maxSlopeDegrees, terrain.slopeDegrees(at: p) > maxSlope { continue }
                let key = UInt64(UInt32(truncatingIfNeeded: ix)) << 32 | UInt64(UInt32(truncatingIfNeeded: iy))
                let plant = ScatterIndividual.make(rule, species: si, at: p, ground: ground, clump: clump,
                                                   key: key, rng: &rng)
                if rule.edgeMargin > 0, regions.contains(where: {
                    ScatterGeometry.contains($0, p) && ScatterGeometry.distanceToOutline($0, p) < rule.edgeMargin
                }) { continue }
                guard mask.isEmpty || mask.isClear(p, radius: plant.radius * 0.7) else { continue }
                // Inside a drift, canopies interlock (a clump reads as one mass, not a grid of islands).
                candidates.append(Candidate(point: plant, priority: priority, tight: 1 - 0.35 * clump.weight))
            }
        }
        // Blue noise: accept in priority order (ties broken by position — a total order).
        candidates.sort { a, b in
            a.priority != b.priority ? a.priority < b.priority
                : (a.point.position.y, a.point.position.x) < (b.point.position.y, b.point.position.x)
        }
        let maxR = candidates.map(\.point.radius).max() ?? 0.1
        var index = ScatterDiscIndex(cell: max(0.1, 2 * maxR * rule.canopyOverlap + rule.minSpacing))
        var out: [ScatterPoint] = []
        for c in candidates {
            let r = c.point.radius
            let overlap = rule.canopyOverlap * c.tight
            let law: (Double) -> Double = { (r + $0) * overlap + rule.minSpacing }
            if index.conflicts(c.point.position, reach: law(maxR), distance: law) { continue }
            index.insert(c.point.position, radius: r)
            out.append(c.point)
        }
        // Stable output order: north-to-south rows, west to east.
        out.sort { ($0.position.y, $0.position.x) < ($1.position.y, $1.position.x) }
        return out
    }
}
