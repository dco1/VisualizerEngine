import Foundation
import simd
import VisualizerMaterials

/// **The Christmas tree** (DH-0489) — a potted fir, its baubles, and a string of lights.
///
/// **Why this is not leaf cards.** Every other `PlantStyle` is built from `PlantLeafCard` blades on
/// a phyllotaxis spiral, which is right for a broadleaf houseplant and wrong for a conifer: a fir's
/// silhouette is a SOLID stacked canopy, and a porcupine of needle cards reads as exactly that (the
/// Visualizer UFO scene's `PineGeometry` banned radiating branch-tip spikes for this reason and
/// built noise-modulated layered cones instead). This file is that idea rebuilt on `Mesh3`: a stack
/// of closed, drooping bough TIERS, each a skirt whose rim is scalloped into bough tips.
///
/// **One envelope, four consumers.** The tier stack (`ConiferTier`) is built ONCE per call and every
/// dependent part reads it rather than re-deriving a radius: the canopy mesh, the baubles hung under
/// the bough tips, the bulbs wound onto the tiers' outer bands, and — through `ConiferEnvelope` —
/// the light the string throws (`FixtureEmission.stringLights`). Move the envelope and the baubles,
/// bulbs and glow all move with it ([[feedback-single-source-of-truth]]).
///
/// Local space as `PottedPlantMesh`: X/Z plan about the trunk axis, Y up from the pot base.
extension PottedPlantMesh {

    // MARK: - The envelope + tiers (the SHARED conifer builder)

    /// The tree is built by `ConiferBuilder` — the one pine generator the engine shares with every
    /// other consuming app — with its `.christmasTree` profile. These aliases keep the plant's
    /// vocabulary (and the tests') stable; nothing here re-derives a tier.
    public typealias ConiferTier = ConiferBuilder.Tier

    /// The canopy's overall cone, sized off the plant's one size lever (`plantSize`).
    public struct ConiferEnvelope {
        public let shared: ConiferBuilder.Envelope
        public var bottomY: Double { shared.bottomY }
        public var apexY: Double { shared.apexY }
        public var baseRadius: Double { shared.baseRadius }

        public init(_ params: some PottedPlantGeometry, soilY: Double) {
            let p = ConiferBuilder.Profile.of(.christmasTree)
            // The STAR's tip is the top of the plant, so `plantSize` stays the whole height above
            // the soil (`PottedPlantParams.totalHeight`, the halo, the stacking role all read it).
            shared = ConiferBuilder.Envelope(
                bottomY: soilY + params.plantSize * p.canopyBaseFrac,   // a hand of bare trunk above the soil
                apexY: soilY + params.plantSize - PottedPlantMesh.coniferStarRise(size: params.plantSize),
                baseRadius: params.plantSize * 0.34,                    // a 1.4 m fir is ~0.95 m across its skirt
                convexity: p.convexity)
        }
        public var height: Double { shared.height }
        public func y(atFraction h: Double) -> Double { shared.y(atFraction: h) }
        public func radius(atFraction h: Double) -> Double { shared.radius(atFraction: h) }
    }

    /// Where the string's single stand-in light sits and how big it is (its `softRadius`). Read by
    /// `FixtureEmission.stringLights`; a pure function of the params, so it needs no RNG.
    public static func stringLightGlow(params: some PottedPlantGeometry) -> (centre: Vec3, radius: Double) {
        let env = ConiferEnvelope(params, soilY: vesselMouthY(params))
        let h = 0.40
        return (Vec3(0, env.y(atFraction: h), 0), env.radius(atFraction: h) * 0.9)
    }

    /// Tier count rides `foliageDensity` (the Fullness slider): 5 sparse → 9 lush.
    public static func coniferTierCount(_ params: some PottedPlantGeometry) -> Int {
        min(9, max(5, Int((4 + 2.5 * Double(params.foliageDensity)).rounded())))
    }

    public static func coniferTiers(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix) -> [ConiferTier] {
        ConiferBuilder.tiers(envelope: ConiferEnvelope(params, soilY: soilY).shared,
                             profile: .of(.christmasTree), count: coniferTierCount(params), rng: &rng)
    }

    /// The needle canopy — `ConiferBuilder.canopy`, the same skirts every conifer is made of.
    public static func coniferCanopy(_ tiers: [ConiferTier]) -> Mesh3 { ConiferBuilder.canopy(tiers) }

    // MARK: - Baubles + the star

    /// Bauble colours: a deep red, an old gold, a champagne silver. Taste — neutral classic
    /// defaults, flagged for Danny.
    public static let baublePalette: [Vec3] = [Vec3(0.50, 0.03, 0.04), Vec3(0.72, 0.52, 0.16), Vec3(0.70, 0.68, 0.64)]

    /// Baubles hung just under randomly chosen bough tips (the droop is where a real one hangs),
    /// plus a gold star on the leader. One `ColoredGroup` per palette colour, never one per bauble.
    public static func coniferBaubles(_ tiers: [ConiferTier], params: some PottedPlantGeometry,
                               rng: inout SplitMix) -> [ColoredGroup] {
        var groups = baublePalette.map { ColoredGroup(color: $0, mesh: Mesh3()) }
        let radius = 0.024 + 0.01 * params.plantSize
        for t in tiers {
            let count = max(1, Int((Double(t.lobes) * 0.4 * Double(params.foliageDensity)).rounded()))
            var tips = Array(0 ..< t.lobes)
            for i in 0 ..< min(count, tips.count) {                  // partial Fisher-Yates
                let j = i + Int(rng.unit() * Double(tips.count - i)) % (tips.count - i)
                tips.swapAt(i, j)
                let theta = t.phase + 2 * .pi * Double(tips[i]) / Double(t.lobes)
                let r = t.rimRadius(theta) * 0.90
                let centre = Vec3(cos(theta) * r, t.rimY(theta) - radius * 1.05, sin(theta) * r)
                let c = Int(rng.unit() * Double(baublePalette.count)) % baublePalette.count
                groups[c].mesh.append(coniferBall(centre, radius: radius, segments: 12))
            }
        }
        if let tip = tiers.last {
            groups[1].mesh.append(coniferStar(apexY: tip.yTop, size: params.plantSize))
        }
        return groups.filter { !$0.mesh.isEmpty }
    }

    public static func coniferStarOuterRadius(size: Double) -> Double { 0.05 + 0.02 * size }
    /// How far the star's tip stands above the canopy's apex — read by `ConiferEnvelope` so the
    /// star, not the needles, is what reaches `plantSize`.
    public static func coniferStarRise(size: Double) -> Double { coniferStarOuterRadius(size: size) * 1.85 }

    /// A faceted five-point star standing on the leader: a closed bipyramid over a star outline.
    public static func coniferStar(apexY: Double, size: Double) -> Mesh3 {
        let outer = coniferStarOuterRadius(size: size)
        let inner = outer * 0.42
        let depth = outer * 0.28
        let centre = Vec3(0, apexY + outer * 0.85, 0)
        let outline = (0 ..< 10).map { k -> Vec3 in
            let a = Double.pi / 2 + Double(k) * .pi / 5
            let r = k % 2 == 0 ? outer : inner
            return Vec3(cos(a) * r, centre.y + sin(a) * r, 0)
        }
        var m = Mesh3()
        for k in 0 ..< 10 {
            let a = outline[k], b = outline[(k + 1) % 10]
            for apex in [Vec3(0, centre.y, depth), Vec3(0, centre.y, -depth)] {
                let centroid = Vec3((a.x + b.x + apex.x) / 3, (a.y + b.y + apex.y) / 3, (a.z + b.z + apex.z) / 3)
                m.addTriangle(a, b, apex, outward: centroid - centre)
            }
        }
        return m
    }

    // MARK: - The string of lights

    /// The string, as ONE continuous helix wound onto the tree — a turn per tier, climbing from each
    /// tier's rim band into the next — with a bulb every ~16 cm and a THIN WIRE tracing the whole
    /// route between them. Bulbs stay in the OUTER band of each tier (`u` 0.70…0.95), the part a
    /// person can see: further in, the next tier's skirt hangs over them.
    ///
    /// The wire is built from the SAME samples as the bulbs (it interpolates the bulbs' own
    /// `(azimuth, u)` on the tier surface), so it hugs the bough the bulbs sit on and passes
    /// through every bulb — one route, two meshes ([[feedback-single-source-of-truth]]).
    public static func coniferStringLights(_ tiers: [ConiferTier], params: some PottedPlantGeometry,
                                           rng: inout SplitMix) -> (bulbs: Mesh3, wire: Mesh3) {
        var bulbs = Mesh3()
        let radius = 0.0065 + 0.002 * params.plantSize
        let spacing = 0.16                        // metres of rim between two bulbs
        var theta = rng.unit() * 2 * .pi
        // The route: one point per bulb, plus the tier it rides.
        var route: [(tier: Int, a: Double, u: Double)] = []
        for (ti, t) in tiers.enumerated() {
            let n = max(3, Int((2 * .pi * t.rRim / spacing).rounded()))
            for j in 0 ..< n {
                let f = Double(j) / Double(n)
                let a = theta + 2 * .pi * f + (rng.unit() - 0.5) * 0.12
                let u = min(1, max(0, 0.95 - 0.25 * f + (rng.unit() - 0.5) * 0.06))
                route.append((ti, a, u))
            }
            theta += 2 * .pi
        }
        func lift(_ tier: Int, _ a: Double, _ u: Double) -> Vec3 {
            let p = tiers[tier].top(u, a)
            // Proud of the needles: out along the radius and up, so a bulb sits ON the bough.
            return Vec3(p.x + cos(a) * radius * 0.4, p.y + radius * 0.6, p.z + sin(a) * radius * 0.4)
        }
        for r in route { bulbs.append(coniferBall(lift(r.tier, r.a, r.u), radius: radius, segments: 8)) }

        // Wire: densify each leg (same tier → walk the tier's own surface; tier→tier → a short sag).
        var path: [Vec3] = []
        let steps = 4
        for k in 0 ..< route.count {
            let r = route[k]
            path.append(lift(r.tier, r.a, r.u))
            guard k + 1 < route.count else { break }
            let n = route[k + 1]
            for s in 1 ..< steps {
                let f = Double(s) / Double(steps)
                if n.tier == r.tier {
                    path.append(lift(r.tier, r.a + (n.a - r.a) * f, r.u + (n.u - r.u) * f))
                } else {
                    let p0 = lift(r.tier, r.a, r.u), p1 = lift(n.tier, n.a, n.u)
                    let sag = sin(f * .pi) * 0.012
                    path.append(Vec3(p0.x + (p1.x - p0.x) * f, p0.y + (p1.y - p0.y) * f - sag,
                                     p0.z + (p1.z - p0.z) * f))
                }
            }
        }
        var wire = Mesh3()
        if path.count >= 2 {
            let wr = 0.0011 + 0.0004 * params.plantSize
            wire.sweep(profile: .circle(radius: wr, segments: 4), along: path, capStart: true, capEnd: true)
        }
        return (bulbs, wire)
    }

    /// The bulbs alone (kept for callers that only want the glass).
    public static func coniferBulbs(_ tiers: [ConiferTier], params: some PottedPlantGeometry, rng: inout SplitMix) -> Mesh3 {
        coniferStringLights(tiers, params: params, rng: &rng).bulbs
    }

    /// A small closed ball: one pole-to-pole `revolve` (no internal caps), smoothed. The tree carries
    /// ~100 bulbs and ~30 baubles, so their segment count is a triangle budget, not a chord-error
    /// target — at 8 segments a 9 mm bulb is 48 triangles, and it is a few pixels across.
    public static func coniferBall(_ centre: Vec3, radius: Double, segments: Int) -> Mesh3 {
        let bands = max(3, segments / 2)
        let profile = (0 ... bands).map { i -> Mesh3.ProfilePoint in
            let a = (Double(i) / Double(bands) - 0.5) * .pi
            return .init(r: i == 0 || i == bands ? 0 : cos(a) * radius, y: sin(a) * radius)
        }
        var m = Mesh3()
        m.revolve(profile: profile, segments: segments, capBottom: false, capTop: false)
        return m.smoothed(creaseDegrees: 90).translated(by: centre)
    }

    /// Everything above in one pass, in a fixed RNG order so a seed is a tree.
    public static func christmasTree(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix)
        -> (canopy: Mesh3, baubles: [ColoredGroup], bulbs: Mesh3, wire: Mesh3) {
        let tiers = coniferTiers(params: params, soilY: soilY, rng: &rng)
        let canopy = coniferCanopy(tiers)
        let baubles = coniferBaubles(tiers, params: params, rng: &rng)
        let lights = coniferStringLights(tiers, params: params, rng: &rng)
        return (canopy, baubles, lights.bulbs, lights.wire)
    }
}
