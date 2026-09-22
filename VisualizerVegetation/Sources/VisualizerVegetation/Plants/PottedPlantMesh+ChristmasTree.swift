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

    // MARK: - The envelope

    /// The canopy's overall cone — where it starts, where the tip is, how wide the base tier is.
    /// Everything here is a fraction of `plantSize`, the style's one size lever.
    public struct ConiferEnvelope {
        public let bottomY: Double
        public let apexY: Double
        public let baseRadius: Double

        public init(_ params: some PottedPlantGeometry, soilY: Double) {
            bottomY = soilY + params.plantSize * 0.10   // a hand of bare trunk above the soil
            // The STAR's tip is the top of the plant, so `plantSize` stays the whole height above
            // the soil (`PottedPlantParams.totalHeight`, the halo, the stacking role all read it).
            apexY = soilY + params.plantSize - PottedPlantMesh.coniferStarRise(size: params.plantSize)
            baseRadius = params.plantSize * 0.34        // a 1.4 m fir is ~0.95 m across its skirt
        }

        public var height: Double { apexY - bottomY }
        public func y(atFraction h: Double) -> Double { bottomY + height * h }
        /// Slightly convex cone: a real fir holds its width a little further up than a straight cone.
        public func radius(atFraction h: Double) -> Double { baseRadius * pow(max(0, 1 - h), 0.95) }
    }

    /// Where the string's single stand-in light sits and how big it is (its `softRadius`). Read by
    /// `FixtureEmission.stringLights`; a pure function of the params, so it needs no RNG.
    public static func stringLightGlow(params: some PottedPlantGeometry) -> (centre: Vec3, radius: Double) {
        let env = ConiferEnvelope(params, soilY: vesselMouthY(params))
        let h = 0.40
        return (Vec3(0, env.y(atFraction: h), 0), env.radius(atFraction: h) * 0.9)
    }

    // MARK: - The tiers

    /// One bough tier: a closed skirt. Its TOP surface falls from the trunk to a scalloped rim with
    /// a gravity droop that grows toward the tips; its UNDERSIDE climbs back in to the trunk; a short
    /// inner wall closes it. Watertight, so the canopy is a set of solids, not a cardboard shell.
    public struct ConiferTier {
        public var yTop: Double          // where the top surface meets the trunk
        public var yRim: Double          // rim height before the tips droop
        public var yUnder: Double        // where the underside meets the trunk
        public var rInner: Double        // top surface's inner radius (0 = the tree's tip)
        public var rUnderInner: Double   // underside's inner radius
        public var rRim: Double          // rim radius at a bough TIP
        public var lobes: Int            // bough tips around the rim
        public var phase: Double
        public var noiseA: Double
        public var noiseB: Double
        public var droop: Double         // how far a bough tip hangs below the rim

        /// 1 at a bough tip, 0 in the notch between two — sharp tips, broad notches.
        public func lobe(_ theta: Double) -> Double {
            pow(0.5 + 0.5 * cos(Double(lobes) * (theta - phase)), 2.5)
        }
        /// Seeded low-frequency wobble so no two tiers (and no two trees) share a rim.
        public func noise(_ theta: Double) -> Double {
            0.6 * sin(3 * theta + noiseA) + 0.4 * sin(7 * theta + noiseB)
        }
        public func rimRadius(_ theta: Double) -> Double {
            rRim * (0.78 + 0.22 * lobe(theta)) * (1 + 0.05 * noise(theta))
        }
        public func rimY(_ theta: Double) -> Double { yRim - droop * lobe(theta) }

        /// A point on the top surface, `u` 0 at the trunk → 1 at the rim. `pow(u, 1.3)` keeps the
        /// bough shallow near the trunk and steep at the tip — it droops, it does not tent.
        public func top(_ u: Double, _ theta: Double) -> Vec3 {
            let r = rInner + (rimRadius(theta) - rInner) * u
            let y = yTop - (yTop - yRim) * pow(u, 1.3) - droop * lobe(theta) * u * u
            return Vec3(cos(theta) * r, y, sin(theta) * r)
        }
        /// A point on the underside, `v` 0 at the rim → 1 at the trunk.
        public func under(_ v: Double, _ theta: Double) -> Vec3 {
            let r = rimRadius(theta) + (rUnderInner - rimRadius(theta)) * v
            let y = rimY(theta) + (yUnder - rimY(theta)) * pow(v, 0.7)
            return Vec3(cos(theta) * r, y, sin(theta) * r)
        }
    }

    /// Tier count rides `foliageDensity` (the Fullness slider): 5 sparse → 9 lush.
    public static func coniferTierCount(_ params: some PottedPlantGeometry) -> Int {
        min(9, max(5, Int((4 + 2.5 * Double(params.foliageDensity)).rounded())))
    }

    public static func coniferTiers(params: some PottedPlantGeometry, soilY: Double, rng: inout SplitMix) -> [ConiferTier] {
        let env = ConiferEnvelope(params, soilY: soilY)
        let n = coniferTierCount(params)
        let reach = 0.86                          // the tier RIMS span this much of the canopy
        let spacing = reach / Double(n)
        return (0 ..< n).map { i in
            let hb = Double(i) * spacing
            let isTip = i == n - 1
            // Each tier's top reaches 1.6 rim-spacings up, well past the next tier's rim — the
            // overlap is what makes the stack read as layered boughs rather than a pagoda of discs.
            let ht = isTip ? 1.0 : min(1.0, hb + 1.6 * spacing)
            let rRim = max(0.04, env.radius(atFraction: hb)) * (0.94 + 0.12 * rng.unit())
            let yRim = env.y(atFraction: hb)
            let yTop = env.y(atFraction: ht)
            let inner = max(0.02, rRim * 0.10)
            return ConiferTier(yTop: yTop, yRim: yRim, yUnder: yRim + (yTop - yRim) * 0.30,
                               rInner: isTip ? 0 : inner, rUnderInner: inner,
                               rRim: rRim, lobes: max(5, Int((rRim / 0.055).rounded())),
                               phase: rng.unit() * 2 * .pi,
                               noiseA: rng.unit() * 2 * .pi, noiseB: rng.unit() * 2 * .pi,
                               droop: rRim * 0.10)
        }
    }

    // MARK: - The canopy

    /// Azimuth segments per tier — enough for ~5 per bough tip on the widest tier.
    public static let coniferSegments = 44

    /// The needle canopy: every tier as a closed, smoothed skirt. Smoothed PER TIER (tiers overlap,
    /// and averaging normals across two tiers' coincident points would blend unrelated surfaces);
    /// the 40° crease keeps each rim a crisp edge.
    public static func coniferCanopy(_ tiers: [ConiferTier]) -> Mesh3 {
        let s = coniferSegments
        let thetas = (0 ..< s).map { Double($0) / Double(s) * 2 * .pi }
        var canopy = Mesh3()
        for t in tiers {
            // Rings, inside → out → back in: top surface u = 0…1, then the underside v = ⅓…1
            // (v = 0 IS the rim, already the top surface's last ring — shared, so the seam welds).
            let topRings = [0.0, 0.42, 0.76, 1.0].map { u in thetas.map { t.top(u, $0) } }
            let underRings = [0.45, 1.0].map { v in thetas.map { t.under(v, $0) } }
            var m = Mesh3()
            func strip(_ a: [Vec3], _ b: [Vec3], outward: (Double) -> Vec3) {
                for j in 0 ..< s {
                    let k = (j + 1) % s
                    let mid = (thetas[j] + (k == 0 ? 2 * .pi : thetas[k])) / 2
                    m.addQuad(a[j], a[k], b[k], b[j], outward: outward(mid))
                }
            }
            // Top surface faces up-and-out; the underside down-and-in; the inner wall toward the axis.
            for r in 0 ..< topRings.count - 1 {
                strip(topRings[r], topRings[r + 1]) { Vec3(cos($0), 1, sin($0)) }
            }
            strip(topRings[topRings.count - 1], underRings[0]) { Vec3(-0.3 * cos($0), -1, -0.3 * sin($0)) }
            strip(underRings[0], underRings[1]) { Vec3(-0.3 * cos($0), -1, -0.3 * sin($0)) }
            strip(underRings[1], topRings[0]) { Vec3(-cos($0), -0.2, -sin($0)) }
            canopy.append(m.smoothed(creaseDegrees: 40))
        }
        return canopy
    }

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

    /// Bulbs wound onto the tree as ONE continuous helix — a turn per tier, climbing from each
    /// tier's rim band into the next — and kept to the OUTER band of each tier (`u` 0.70…0.95),
    /// which is the part a person can see: further in, the next tier's skirt hangs over them.
    public static func coniferBulbs(_ tiers: [ConiferTier], params: some PottedPlantGeometry, rng: inout SplitMix) -> Mesh3 {
        var m = Mesh3()
        let radius = 0.0065 + 0.002 * params.plantSize
        let spacing = 0.16                        // metres of rim between two bulbs
        var theta = rng.unit() * 2 * .pi
        for t in tiers {
            let n = max(3, Int((2 * .pi * t.rRim / spacing).rounded()))
            for j in 0 ..< n {
                let f = Double(j) / Double(n)
                let a = theta + 2 * .pi * f + (rng.unit() - 0.5) * 0.12
                let u = 0.95 - 0.25 * f + (rng.unit() - 0.5) * 0.06
                let p = t.top(min(1, max(0, u)), a)
                // Proud of the needles: out along the radius and up, so a bulb sits ON the bough.
                let centre = Vec3(p.x + cos(a) * radius * 0.4, p.y + radius * 0.6, p.z + sin(a) * radius * 0.4)
                m.append(coniferBall(centre, radius: radius, segments: 8))
            }
            theta += 2 * .pi
        }
        return m
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
        -> (canopy: Mesh3, baubles: [ColoredGroup], bulbs: Mesh3) {
        let tiers = coniferTiers(params: params, soilY: soilY, rng: &rng)
        let canopy = coniferCanopy(tiers)
        let baubles = coniferBaubles(tiers, params: params, rng: &rng)
        let bulbs = coniferBulbs(tiers, params: params, rng: &rng)
        return (canopy, baubles, bulbs)
    }
}
