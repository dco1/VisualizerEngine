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
        groups[1].mesh.append(coniferStar(apexY: Double(sk.apexY), size: params.plantSize))
        // The garland is its OWN group: gold TINSEL is a satin metal, not the mirror glass of a bauble.
        groups.append(ColoredGroup(color: Vec3(0.78, 0.56, 0.18), mesh: coniferGarland(sk, params: params),
                                   metallic: 0.85, roughness: 0.30))
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

    /// **The garland** — ONE continuous strand of gold tinsel, spiralling down the tree from the top and
    /// draped between the branches. An earlier version drew a separate little arc between each pair of
    /// neighbouring branch tips, hung in the air outside them, as a smooth rod: every swag had two loose
    /// ends and none of it touched the tree. This is a single path (`coniferGarlandPath`), so there is
    /// nothing to be unconnected: it starts tucked into the tree beside the leader, winds down ~one turn
    /// per 24 cm of canopy, sags between a support every quarter-turn (where it rests on the branch
    /// tips) and dips a little into the foliage at mid-span, and ends in a short free tail.
    public static func coniferGarlandPath(_ sk: FirSkeleton, params: some PottedPlantGeometry) -> [Vec3] {
        var rng = SplitMix(params.seed &* 0xA24BAED4963EE407 &+ 0x2545F4914F6CDD1D)
        let H = Double(sk.height)
        let turns = max(4.0, (H / 0.24).rounded())
        let support = Double.pi / 2                          // a support every quarter-turn
        let theta0 = rng.unit() * 2 * .pi
        let soil = Double(sk.bottomY) - params.plantSize * 0.10
        // Each span sags its own amount (deterministic per span) so the swags are not a repeated pattern.
        let spans = Int(turns * 4) + 2
        let sagAmp: [Double] = (0 ..< spans).map { _ in (0.040 + 0.030 * rng.unit()) * H / 1.2 }
        func smooth(_ x: Double) -> Double { let t = max(0, min(1, x)); return t * t * (3 - 2 * t) }
        func at(_ u: Double) -> Vec3 {
            let f = 0.93 - 0.82 * u                           // from near the leader down to the lowest whorls
            let th = theta0 + turns * 2 * .pi * u
            let span = th / support
            let local = span - floor(span)
            let sag = sagAmp[min(spans - 1, max(0, Int(u * Double(spans - 1))))] * sin(.pi * local)
            var rho = Double(sk.radius(atFraction: Float(f))) * 0.985 - 0.014 * sin(.pi * local)
            var y = Double(sk.y(atFraction: Float(f))) - sag
            if u < 0.05 { rho *= 0.25 + 0.75 * smooth(u / 0.05) }                    // tucked in at the top
            if u > 0.97 { y -= (u - 0.97) / 0.03 * 0.07 }                            // a free tail at the bottom
            return Vec3(cos(th) * rho, max(y, soil + 0.035), sin(th) * rho)
        }
        // Dense sample, then resample at even arc length (the rope's twist is per metre of strand).
        let dense = (0 ... Int(turns) * 260).map { at(Double($0) / Double(Int(turns) * 260)) }
        var out: [Vec3] = [dense[0]]
        var carry = 0.0
        let step = 0.015
        for i in 1 ..< dense.count {
            var a = dense[i - 1]
            let seg = len3(dense[i] - a)
            var left = seg
            while carry + left >= step {
                let t = (step - carry) / left
                a = a + (dense[i] - a) * t
                out.append(a)
                left = len3(dense[i] - a); carry = 0
            }
            carry += left
        }
        out.append(dense[dense.count - 1])
        return out
    }

    /// A TWISTED TINSEL ROPE along `path`: a fluted cross-section (alternating high and low ridges) that
    /// rotates as it travels, which is what makes a tinsel garland read as a spun, glittering strand
    /// rather than a smooth tube. Flat-shaded on purpose (each facet catches the light separately — that
    /// IS the glitter). Closed solid with fan caps.
    public static func twistedRope(along path: [Vec3], radius: Double, lobes: Int = 4,
                                   twistsPerMetre: Double = 8) -> Mesh3 {
        var m = Mesh3()
        guard path.count >= 2 else { return m }
        let N = lobes * 2
        // Parallel-transport frames.
        var tangents: [Vec3] = path.indices.map { i in
            let d = i == 0 ? path[1] - path[0] : (i == path.count - 1 ? path[i] - path[i - 1] : path[i + 1] - path[i - 1])
            return len3(d) > 1e-9 ? normalize3(d) : Vec3(0, 1, 0)
        }
        var us: [Vec3] = [], vs: [Vec3] = []
        var u0 = cross3(tangents[0], abs(tangents[0].y) < 0.95 ? Vec3(0, 1, 0) : Vec3(1, 0, 0))
        u0 = normalize3(u0)
        us.append(u0); vs.append(cross3(tangents[0], u0))
        for i in 1 ..< path.count {
            // Rotate the previous u by the minimal rotation taking tangent[i-1] → tangent[i].
            let a = tangents[i - 1], b = tangents[i]
            let axis = cross3(a, b), s = len3(axis), c = dot3(a, b)
            var u = us[i - 1]
            if s > 1e-9 {
                let k = axis / s, ang = atan2(s, c)
                u = u * cos(ang) + cross3(k, u) * sin(ang) + k * (dot3(k, u) * (1 - cos(ang)))
            }
            u = normalize3(u - tangents[i] * dot3(u, tangents[i]))
            us.append(u); vs.append(cross3(tangents[i], u))
        }
        tangents = tangents.map { $0 }
        var arc = 0.0
        var rings: [[Vec3]] = []
        for i in path.indices {
            if i > 0 { arc += len3(path[i] - path[i - 1]) }
            let twist = 2 * Double.pi * twistsPerMetre * arc / Double(lobes)
            rings.append((0 ..< N).map { k in
                let ang = 2 * .pi * Double(k) / Double(N) + twist
                let r = radius * (k % 2 == 0 ? 1.0 : 0.52)
                return path[i] + (us[i] * cos(ang) + vs[i] * sin(ang)) * r
            })
        }
        for i in 0 ..< path.count - 1 {
            for k in 0 ..< N {
                let k2 = (k + 1) % N
                let a = rings[i][k], b = rings[i][k2], c = rings[i + 1][k2], d = rings[i + 1][k]
                let centre = (a + b + c + d) * 0.25 - (path[i] + path[i + 1]) * 0.5
                m.addQuad(a, b, c, d, outward: centre)
            }
        }
        // End caps.
        for (i, outward) in [(0, tangents[0] * -1), (path.count - 1, tangents[path.count - 1])] {
            for k in 0 ..< N {
                m.addTriangle(path[i], rings[i][k], rings[i][(k + 1) % N], outward: outward)
            }
        }
        return m
    }

    /// The garland's core: a thin twisted gold rope along the continuous path (a closed solid).
    public static func coniferGarlandRope(_ sk: FirSkeleton, params: some PottedPlantGeometry) -> Mesh3 {
        twistedRope(along: coniferGarlandPath(sk, params: params), radius: garlandCoreRadius(params))
    }

    public static func garlandCoreRadius(_ params: some PottedPlantGeometry) -> Double { 0.0058 + 0.0010 * params.plantSize }

    /// **Tinsel is a core wrapped in bristles.** A rope alone reads as a ribbon; the fuzz of fine foil
    /// strands thrown out all round it is what makes it tinsel. Each bristle is one thin double-sided
    /// spike (base ~2.5 mm, 1.2–2.4 cm long, thrown out radially with a little forward sweep and droop),
    /// six per 1.5 cm of strand. Open cards (like leaves), so the strand's closed core is audited
    /// separately from its fuzz.
    public static func tinselBristles(along path: [Vec3], coreRadius: Double, seed: UInt64) -> Mesh3 {
        var m = Mesh3()
        var rng = SplitMix(seed &* 0x9E3779B97F4A7C15 &+ 0x51ED270B)
        for i in 0 ..< path.count - 1 {
            let a = path[i], b = path[i + 1]
            let t = len3(b - a) > 1e-9 ? normalize3(b - a) : Vec3(0, 1, 0)
            var u = cross3(t, abs(t.y) < 0.95 ? Vec3(0, 1, 0) : Vec3(1, 0, 0))
            u = normalize3(u)
            let v = cross3(t, u)
            for _ in 0 ..< 6 {
                let p = a + (b - a) * rng.unit()
                let phi = rng.unit() * 2 * .pi
                var d = u * cos(phi) + v * sin(phi)
                d = normalize3(d + t * ((rng.unit() - 0.5) * 0.9))
                let len = 0.012 + 0.012 * rng.unit()
                let base = p + d * (coreRadius * 0.75)
                var tip = base + d * len
                tip.y -= 0.18 * len                                     // foil droops a little
                let w = normalize3(cross3(d, t)) * 0.00125
                let n = normalize3(cross3(w, tip - base))
                m.addTriangle(base + w, base - w, tip, outward: n)
                m.addTriangle(base + w, base - w, tip, outward: n * -1)
            }
        }
        return m
    }

    /// The whole garland: rope + bristles, one mesh.
    public static func coniferGarland(_ sk: FirSkeleton, params: some PottedPlantGeometry) -> Mesh3 {
        let path = coniferGarlandPath(sk, params: params)
        var m = twistedRope(along: path, radius: garlandCoreRadius(params))
        m.append(tinselBristles(along: path, coreRadius: garlandCoreRadius(params), seed: params.seed))
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
