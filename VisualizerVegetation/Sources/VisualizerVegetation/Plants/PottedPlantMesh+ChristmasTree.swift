import Foundation
import simd
import VisualizerMaterials

/// **The Christmas tree** (DH-0489) — a potted fir, its baubles, a gold garland, and a string of lights.
///
/// **The tree is the unified tree factory's.** `ForestTreeGeometry` grows it (species `.fir`, the
/// factory's conifer growth mode: whorls of branches carrying needled sprigs — see
/// `ForestTreeGeometry+Conifer`). An earlier version built the fir here, as a stack of closed skirts;
/// it read as layers of cut fabric and was a second tree generator beside the one the yard trees use.
/// What lives in this file is only what a POTTED, DRESSED tree adds: the pot-relative envelope, and
/// the ornaments, which attach to the factory's own `ConiferSkeleton` — baubles hang from branches
/// that exist, the string winds around the tree they define.
///
/// Local space as `PottedPlantMesh`: X/Z plan about the trunk axis, Y up from the pot base.
extension PottedPlantMesh {

    // MARK: - The envelope

    /// The tree's overall cone, sized off the plant's one size lever (`plantSize`) and the pot.
    public struct ConiferEnvelope {
        public let bottomY: Double
        public let apexY: Double
        public let baseRadius: Double

        public init(_ params: some PottedPlantGeometry, soilY: Double) {
            bottomY = soilY + params.plantSize * 0.10     // a hand of bare trunk above the soil
            // The STAR's tip is the top of the plant, so `plantSize` stays the whole height above
            // the soil (`PottedPlantParams.totalHeight`, the halo, the stacking role all read it).
            apexY = soilY + params.plantSize - PottedPlantMesh.coniferStarRise(size: params.plantSize)
            baseRadius = params.plantSize * 0.34          // a 1.4 m fir is ~0.95 m across its skirt
        }
        public var height: Double { apexY - bottomY }
        public func y(atFraction h: Double) -> Double { bottomY + height * h }
        public func radius(atFraction h: Double) -> Double { baseRadius * pow(max(0, 1 - h), 0.95) }
    }

    /// Where the string's single stand-in light sits and how big it is (its `softRadius`). Read by
    /// `FixtureEmission.stringLights`; a pure function of the params, so it needs no RNG.
    public static func stringLightGlow(params: some PottedPlantGeometry) -> (centre: Vec3, radius: Double) {
        let env = ConiferEnvelope(params, soilY: vesselMouthY(params))
        let h = 0.40
        return (Vec3(0, env.y(atFraction: h), 0), env.radius(atFraction: h) * 0.9)
    }

    // MARK: - The factory's tree

    public typealias FirSkeleton = ForestTreeGeometry.ConiferSkeleton

    /// The skeleton the potted fir is grown from — the SAME `growConiferSkeleton` the yard trees
    /// use, sized to this plant's envelope. Everything on the tree reads it.
    public static func coniferSkeleton(params: some PottedPlantGeometry, soilY: Double) -> FirSkeleton {
        let env = ConiferEnvelope(params, soilY: soilY)
        let look = ForestTreeGeometry.look(for: .fir)
        return ForestTreeGeometry.growConiferSkeleton(
            bottomY: Float(env.bottomY), apexY: Float(env.apexY), baseRadius: Float(env.baseRadius),
            look: look.conifer!, density: max(0.5, min(2.2, Float(params.foliageDensity))), seed: params.seed)
    }

    /// The whole tree — trunk, branches AND sprigs — as one painted mesh, baked by the factory into a
    /// soup. The wood rides the painted group (with its own bark albedo) rather than the pot's
    /// substrate group, whose fixed earth material would render thin bark as pale tan.
    public static func coniferTreeMeshes(_ sk: FirSkeleton, params: some PottedPlantGeometry, soilY: Double)
        -> (wood: Mesh3, foliage: Mesh3, foliageColors: [Vec3]) {
        let look = ForestTreeGeometry.look(for: .fir)
        var soup = ForestTreeGeometry.Soup()
        soup.skipClearingCull = true          // the Forest scene's clearing cull is not ours (the plant sits at its own origin)
        ForestTreeGeometry.emitConiferWood(&soup, world: matrix_identity_float4x4, sk: sk, look: look,
                                           trunkR: Float(stemRadius(.christmasTree, at: 0)) * Float(max(0.8, params.plantSize / 1.4)),
                                           trunkBaseY: Float(soilY))
        ForestTreeGeometry.emitConiferSprigs(&soup, world: matrix_identity_float4x4, sk: sk, look: look,
                                             seed: params.seed)
        let parts = soup.splitMeshes()
        var all = parts.wood
        all.append(parts.foliage)
        return (Mesh3(), all, parts.woodColors + parts.foliageColors)
    }

    // MARK: - Baubles + the star

    /// Bauble colours: a deep red, an old gold, a champagne silver. Taste — neutral classic
    /// defaults, flagged for Danny.
    public static let baublePalette: [Vec3] = [Vec3(0.50, 0.03, 0.04), Vec3(0.72, 0.52, 0.16), Vec3(0.70, 0.68, 0.64)]

    /// Baubles hung from branches of the lower and middle whorls (the tips are where a real one
    /// hangs), plus the gold star on the leader and the garland. One `ColoredGroup` per palette
    /// colour, never one per bauble.
    public static func coniferBaubles(_ sk: FirSkeleton, params: some PottedPlantGeometry,
                                      rng: inout SplitMix) -> [ColoredGroup] {
        var groups = baublePalette.map { ColoredGroup(color: $0, mesh: Mesh3()) }
        var srng = SplitMix(params.seed &* 0xC6BC279692B5C323 &+ 0x9E3779B97F4A7C15)   // sizes: own stream
        let radius = 0.018 + 0.006 * params.plantSize
        let soil = Double(sk.bottomY) - params.plantSize * 0.10
        let usable = sk.branches.indices.filter { sk.branches[$0].whorl < Int(Double(sk.whorls.count) * 0.82)
                                                  && sk.branches[$0].length > 0.15 }
        let count = min(usable.count, Int(30 * pow(Double(params.foliageDensity), 0.5)))
        var pool = usable
        for i in 0 ..< count {
            let j = i + Int(rng.unit() * Double(pool.count - i)) % (pool.count - i)
            pool.swapAt(i, j)
            let b = sk.branches[pool[i]]
            let p = b.point(at: Float(0.80 + 0.14 * rng.unit()))
            let c = Int(rng.unit() * Double(baublePalette.count)) % baublePalette.count
            let bs = 0.8 + 0.5 * srng.unit()
            let hang = Vec3(Double(p.x), Double(p.y) - (radius * bs * 1.05 + 0.03), Double(p.z))
            let centre = Vec3(hang.x, max(hang.y, soil + 0.03 + radius * bs), hang.z)
            groups[c].mesh.append(coniferBall(centre, radius: radius * bs, segments: 14))
            groups[1].mesh.append(coniferBaubleHardware(centre: centre, radius: radius * bs))
        }
        groups[1].mesh.append(coniferGarland(sk, params: params))      // gold tinsel swags
        groups[1].mesh.append(coniferStar(apexY: Double(sk.apexY), size: params.plantSize))
        return groups.filter { !$0.mesh.isEmpty }
    }

    /// The cap and hook that hang a bauble: a short revolved collar sitting on the ball's neck, and a
    /// thin swept hook rising from it to the bough tip above. Closed solids, so they audit like the
    /// ball does.
    public static func coniferBaubleHardware(centre: Vec3, radius: Double) -> Mesh3 {
        var m = Mesh3()
        let top = centre.y + radius * 0.96
        let capR = radius * 0.30, capH = radius * 0.32
        m.revolve(profile: [.init(r: capR * 0.85, y: top - capH * 0.2), .init(r: capR, y: top + capH * 0.25),
                            .init(r: capR * 0.7, y: top + capH)], segments: 8, capBottom: true, capTop: true)
        // The hook: up from the cap, over a small arc, back down toward the branch.
        let hr = radius * 0.55
        let base = Vec3(centre.x, top + capH, centre.z)
        let path: [Vec3] = (0 ... 4).map { i in
            let a = Double(i) / 4 * 1.35 * .pi
            return Vec3(base.x + (1 - cos(a)) * hr * 0.5, base.y + sin(a) * hr, base.z)
        }
        m.sweep(profile: .circle(radius: radius * 0.07, segments: 4), along: path, capStart: true, capEnd: true)
        return m
    }

    /// **The garland** — a gold tinsel rope draped in swags between the tips of neighbouring branches
    /// in a whorl, on every third whorl. Each swag hangs a little outside the tips at a steady radius
    /// and sags ~13 % of its chord; it rides the gold metallic group with the star and the bauble
    /// hardware.
    public static func coniferGarland(_ sk: FirSkeleton, params: some PottedPlantGeometry) -> Mesh3 {
        var m = Mesh3()
        let r = 0.0035 + 0.0012 * params.plantSize
        let soil = Double(sk.bottomY) - params.plantSize * 0.10
        for (wi, ring) in sk.whorls.enumerated() where wi % 3 == 1 && wi < sk.whorls.count - 1 && ring.count >= 4 {
            for k in 0 ..< ring.count {
                let a = sk.branches[ring[k]], b = sk.branches[ring[(k + 1) % ring.count]]
                let ta = a.point(at: 0.97), tb = b.point(at: 0.97)
                let ra = Double((ta.x * ta.x + ta.z * ta.z).squareRoot()), rb = Double((tb.x * tb.x + tb.z * tb.z).squareRoot())
                let aa = Double(atan2(ta.z, ta.x))
                var bb = Double(atan2(tb.z, tb.x))
                while bb <= aa { bb += 2 * .pi }
                guard bb - aa < 1.5 else { continue }            // only neighbours, not across a gap
                let rad = max(ra, rb) + 0.012
                let chord = rad * (bb - aa)
                let path: [Vec3] = (0 ... 11).map { i in
                    let u = Double(i) / 11
                    let th = aa + u * (bb - aa)
                    let y = Double(ta.y) + Double(tb.y - ta.y) * u - chord * 0.13 * sin(u * .pi) - r
                    return Vec3(cos(th) * rad, max(y, soil + 0.03), sin(th) * rad)
                }
                m.sweep(profile: .circle(radius: r, segments: 5), along: path, capStart: true, capEnd: true)
            }
        }
        return m
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

    /// The string, as ONE continuous helix wound around the tree the skeleton defines — turns every
    /// ~0.2 m of height, a bulb every ~16 cm of wire, sitting just inside the branch tips so the
    /// foliage half-hides it the way a real string is half-hidden — with a THIN WIRE tracing the
    /// whole route between the bulbs. The wire is built from the SAME samples as the bulbs, so it
    /// passes through every one: one route, two meshes ([[feedback-single-source-of-truth]]).
    public static func coniferStringLights(_ sk: FirSkeleton, params: some PottedPlantGeometry,
                                           rng: inout SplitMix) -> (bulbs: Mesh3, wire: Mesh3) {
        var bulbs = Mesh3()
        let radius = 0.0065 + 0.002 * params.plantSize
        let H = Double(sk.height)
        let turns = max(5.0, (H / 0.20).rounded())
        let phase0 = rng.unit() * 2 * .pi
        // Helix position at u in [0,1]: bottom of the foliage to near the top.
        func at(_ u: Double, jitter: Double) -> Vec3 {
            let f = 0.06 + 0.84 * u
            let th = phase0 + turns * 2 * .pi * u
            let rho = Double(sk.radius(atFraction: Float(f))) * (0.95 + 0.10 * jitter)
            return Vec3(cos(th) * rho, Double(sk.y(atFraction: Float(f))) + 0.012, sin(th) * rho)
        }
        // Arc length of the helix, to space the bulbs by distance.
        var length = 0.0, prev = at(0, jitter: 0.5)
        for i in 1 ... 400 { let p = at(Double(i) / 400, jitter: 0.5); length += len3(p - prev); prev = p }
        let n = max(8, Int(length / 0.16))
        var route: [Double] = []
        for k in 0 ... n { route.append(min(1, (Double(k) + (rng.unit() - 0.5) * 0.3) / Double(n))) }
        var centres: [Vec3] = []
        for u in route {
            let c = at(u, jitter: rng.unit())
            centres.append(c)
            bulbs.append(coniferBall(c, radius: radius, segments: 8))
        }
        // Wire: from bulb to bulb along the helix itself (3 sub-steps), with a few mm of sag.
        var path: [Vec3] = []
        for k in 0 ..< route.count {
            path.append(centres[k])
            guard k + 1 < route.count else { break }
            for s in 1 ..< 3 {
                let f = Double(s) / 3
                let u = route[k] + (route[k + 1] - route[k]) * f
                var p = at(u, jitter: 0.5)
                p.y -= 0.006 * sin(f * .pi)
                path.append(p)
            }
        }
        var wire = Mesh3()
        if path.count >= 2 {
            wire.sweep(profile: .circle(radius: 0.0011 + 0.0004 * params.plantSize, segments: 4), along: path,
                       capStart: true, capEnd: true)
        }
        return (bulbs, wire)
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
        -> (foliage: Mesh3, foliageColors: [Vec3], wood: Mesh3, baubles: [ColoredGroup], bulbs: Mesh3, wire: Mesh3) {
        let sk = coniferSkeleton(params: params, soilY: soilY)
        let tree = coniferTreeMeshes(sk, params: params, soilY: soilY)
        let baubles = coniferBaubles(sk, params: params, rng: &rng)
        let lights = coniferStringLights(sk, params: params, rng: &rng)
        return (tree.foliage, tree.foliageColors, tree.wood, baubles, lights.bulbs, lights.wire)
    }
}
