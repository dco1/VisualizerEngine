import Foundation
import simd
import VisualizerMaterials

extension LeafSilhouette {
    /// An aloe leaf: broad where it clasps the stem, then a long even taper to a soft point. The mid
    /// tier is the pups'.
    public static let aloeLeaf = LeafSilhouette(
        name: "aloeLeaf",
        hero: [(0.00, 0.00), (0.04, 0.78), (0.16, 1.00), (0.40, 0.82), (0.64, 0.54), (0.86, 0.24), (1.00, 0.00)],
        mid: [(0.00, 0.00), (0.08, 0.86), (0.40, 0.80), (0.80, 0.30), (1.00, 0.00)])
}

/// *Aloe arborescens* — the candelabra aloe (DH-0797), the aloe of Southern California's gardens
/// and coastal hedges. (*Aloe vera*'s single ground rosette is what a young one looks like; this one
/// goes on to branch.)
///
/// What makes it read as this plant, trait by trait:
///  - **Thick, tapering, recurved leaves** in golden-angle rosettes, arching outward and up; plump
///    in section, channelled above and convex below (`FleshyLeaf`, on the opaque `stem` part).
///  - **Soft pale marginal teeth** down both margins — short and blunt, barely hooked, unlike an
///    agave's. They are the `foliage` part, so the species' `leafColor` is their pale cream-green.
///  - **Solid grey-green**, and a **red-orange blush at the tips** of the sun-struck outer leaves.
///  - **Multiple rosettes on woody stems**: past `aloeBranchingSize` the plant is a shrub; below it,
///    a young single rosette on the ground.
///  - **Tall red-orange racemes** rising well clear of the leaves: a cone of tubular flowers, the
///    open ones pendulous below, tight red buds above.
///  - **Withered lower leaves** drooping below each rosette, dried brown from the tip, while the
///    inner leaves stay upright and plump.
///  - **Basal pups** — small rosettes on the ground beside the plant.
extension GardenPlantMesh {

    public struct AloeRosette: Sendable, Equatable {
        /// The growing point the leaves spiral out of.
        public var centre: Vec3
        public var axis: Vec3
        /// The longest leaf.
        public var leafLength: Double
        /// Where its woody stem leaves the ground — nil when the rosette sits on the ground.
        public var stemFoot: Vec3?
        public var pup: Bool
        public var inFlower: Bool
    }

    public struct AloeLeaf: Sendable, Equatable {
        public var leaf: FleshyLeaf
        public var rosette: Int
        /// 0 the outermost live leaf → 1 the innermost (withered leaves are 0).
        public var rank: Double
        public var withered: Bool
        /// The tip past this fraction wears the sun blush (1 = none).
        public var blushFrom: Double
    }

    public struct AloeRaceme: Sendable, Equatable {
        public var rosette: Int
        public var foot: Vec3
        public var top: Vec3
    }

    public static let aloeWitheredColor = Vec3(0.46, 0.36, 0.24)
    public static let aloeBlushColor    = Vec3(0.66, 0.34, 0.20)
    public static let aloeWoodColor     = Vec3(0.34, 0.30, 0.25)
    public static let aloePeduncleColor = Vec3(0.42, 0.38, 0.24)
    public static let aloeFlowerColor   = Vec3(0.92, 0.32, 0.08)
    public static let aloeBudColor      = Vec3(0.64, 0.12, 0.07)
    public static let aloeLiveLeaves = 10
    public static let aloeWitheredLeaves = 2
    public static let aloePupLeaves = 6
    public static let aloeTeethPerSide = 6
    /// A plant shorter than this is a young single rosette; taller, it has branched into a shrub.
    public static let aloeBranchingSize = 0.45
    public static let aloeFlowersPerRaceme = 22
    /// The flowered fraction of a raceme's stalk, at its top.
    public static let aloeRacemeHead = 0.42
    /// Below this fraction of the head the flowers are open and hang; above it, buds.
    public static let aloeOpenBelow = 0.55
    /// A withered leaf is dried brown from here to its tip.
    public static let aloeWitherFrom = 0.30

    /// The rosettes, the plant's own first and its pups last — the single source for the mesh and
    /// its tests.
    public static func aloeRosettes(spec: GardenPlantSpec) -> [AloeRosette] {
        var rng = speciesRNG(spec, lane: 0xA10E)
        let s = spec.size, density = Double(spec.foliageDensity)
        let up = Vec3(0, 1, 0)
        var out: [AloeRosette] = []
        if s < aloeBranchingSize {
            out.append(AloeRosette(centre: Vec3(0, s * 0.05, 0), axis: up, leafLength: s * 0.78,
                                   stemFoot: nil, pup: false, inFlower: true))
        } else {
            let count = density < 0.8 ? 2 : 3
            let start = rng.unit() * 2 * Double.pi
            for i in 0 ..< count {
                let az = start + Double(i) * 2 * Double.pi / Double(count) + (rng.unit() - 0.5) * 0.7
                let outward = Vec3(cos(az), 0, sin(az))
                let height = s * (i == 0 ? 0.42 : 0.22 + 0.16 * rng.unit())
                let reach = s * (0.16 + 0.16 * rng.unit())
                out.append(AloeRosette(centre: outward * reach + up * height,
                                       axis: normalize3(up + outward * 0.35),
                                       leafLength: s * (0.32 + 0.05 * rng.unit()),
                                       stemFoot: outward * (s * 0.04), pup: false,
                                       inFlower: i == 0 || rng.unit() < 0.6))
            }
        }
        let pups = density < 1.5 ? 1 : 2
        let start = rng.unit() * 2 * Double.pi
        for i in 0 ..< pups {
            let az = start + Double(i) * 2.4
            let r = s * (0.32 + 0.12 * rng.unit())
            out.append(AloeRosette(centre: Vec3(r * cos(az), 0, r * sin(az)), axis: up,
                                   leafLength: s * (s < aloeBranchingSize ? 0.26 : 0.16),
                                   stemFoot: nil, pup: true, inFlower: false))
        }
        return out
    }

    public static func aloeLeaves(spec: GardenPlantSpec, rosettes: [AloeRosette]) -> [AloeLeaf] {
        var rng = speciesRNG(spec, lane: 0xA10F)
        var out: [AloeLeaf] = []
        for (ri, r) in rosettes.enumerated() {
            let phase = rng.unit() * 2 * Double.pi
            let withered = r.pup ? 0 : aloeWitheredLeaves
            let live = r.pup ? aloePupLeaves : aloeLiveLeaves
            for i in 0 ..< withered + live {
                let dead = i < withered
                let t = dead ? 0 : Double(i - withered) / Double(live - 1)
                let radial = rosetteRadial(axis: r.axis, index: i, phase: phase)
                let pitch = dead ? -0.40 - rng.unit() * 0.25 : 0.20 + 1.10 * pow(t, 0.9)
                let frame = rosetteLeafFrame(axis: r.axis, radial: radial, pitch: pitch)
                let length = r.leafLength * (dead ? 0.85 : 1.0 - 0.45 * t) * (0.93 + rng.unit() * 0.14)
                let width = length * 0.13
                let blush = !dead && !r.pup && t < 0.55 && rng.unit() < 0.8
                out.append(AloeLeaf(
                    leaf: FleshyLeaf(base: r.centre + radial * (r.leafLength * 0.03 * (1 - t))
                                        + r.axis * (r.leafLength * (dead ? -0.04 : 0.10 * t)),
                                     yAxis: frame.yAxis, normal: frame.normal, length: length, width: width,
                                     thickness: width * 0.20, crown: 0.6, tipThickness: 0.30, channel: 0.07,
                                     arch: dead ? 0.9 : 0.08 + 0.60 * (1 - t)),
                    rosette: ri, rank: t, withered: dead,
                    blushFrom: blush ? 0.80 + rng.unit() * 0.08 : 1))
            }
        }
        return out
    }

    /// One raceme per flowering rosette, its top standing well clear of the leaves.
    public static func aloeRacemes(spec: GardenPlantSpec, rosettes: [AloeRosette]) -> [AloeRaceme] {
        rosettes.enumerated().compactMap { i, r in
            guard r.inFlower else { return nil }
            let rise = max(spec.size - r.centre.y, r.leafLength * 1.8)
            let lean = Vec3(r.axis.x, 0, r.axis.z) * (rise * 0.25)
            return AloeRaceme(rosette: i, foot: r.centre, top: r.centre + Vec3(0, rise, 0) + lean)
        }
    }

    /// A small smooth revolved solid (a floret tube) stood on `axis` at `origin`.
    private static func aloeFloret(length: Double, radius r: Double, origin: Vec3, axis: Vec3) -> Mesh3 {
        var m = Mesh3()
        guard m.revolve(profile: [Mesh3.ProfilePoint(r: r * 0.45, y: 0), Mesh3.ProfilePoint(r: r, y: length * 0.55),
                                  Mesh3.ProfilePoint(r: r * 0.75, y: length * 0.92), Mesh3.ProfilePoint(r: 0, y: length)],
                        segments: 5) else { return Mesh3() }
        let f = frame(yAxis: axis)
        return m.smoothed(creaseDegrees: 80).placed(xAxis: f.x, yAxis: f.y, zAxis: f.z, origin: origin)
    }

    public static func aloeParts(spec: GardenPlantSpec) -> Parts {
        let rosettes = aloeRosettes(spec: spec)
        let hero = LeafSilhouette.aloeLeaf.margin(tier: .hero)
        let mid = LeafSilhouette.aloeLeaf.margin(tier: .mid)
        var body = Mesh3(), teeth = Mesh3()
        var withered = Mesh3(), blush = Mesh3(), wood = Mesh3(), peduncles = Mesh3(), flowers = Mesh3(), buds = Mesh3()

        for l in aloeLeaves(spec: spec, rosettes: rosettes) {
            let pup = rosettes[l.rosette].pup
            let margin = pup ? mid : hero
            let built = fleshyLeafBuild(l.leaf, silhouette: .aloeLeaf, tier: pup ? .mid : .hero)
            if l.withered {
                let (green, brown) = split(built, at: aloeWitherFrom)
                body.append(green)
                withered.append(brown)
                continue
            }
            let (green, tip) = split(built, at: l.blushFrom)
            body.append(green)
            blush.append(tip)
            if !pup {
                appendMarginTeeth(&teeth, leaf: l.leaf, margin: margin, perSide: aloeTeethPerSide,
                                  from: 0.12, to: 0.88, length: l.leaf.length * 0.02, hook: 0.3, stoutness: 0.45)
            }
        }

        for r in rosettes {
            let core = r.leafLength * 0.07
            body.append(revolvedPart([(core, -core), (core * 0.9, core * 0.8), (0, core * 1.4)],
                                     segments: 8, origin: r.centre, axis: r.axis))
            guard let foot = r.stemFoot else { continue }
            // The woody stem: from the ground, bowing out, into the rosette's core.
            let bow = foot + (r.centre - foot) * 0.5 + Vec3(r.axis.x, 0, r.axis.z) * (spec.size * 0.03)
            var stem = Mesh3()
            if stem.sweep(profile: .circle(radius: spec.size * 0.026, segments: 6),
                          along: [foot - Vec3(0, spec.size * 0.03, 0), bow, r.centre],
                          scales: [1.2, 1.0, 0.8], capStart: false, capEnd: false) {
                wood.append(stem.smoothed())
            }
        }

        var rng = speciesRNG(spec, lane: 0xA110)
        for raceme in aloeRacemes(spec: spec, rosettes: rosettes) {
            let axis = normalize3(raceme.top - raceme.foot)
            var stalk = Mesh3()
            let mid = raceme.foot + (raceme.top - raceme.foot) * 0.5
            if stalk.sweep(profile: .circle(radius: spec.size * 0.008, segments: 5),
                           along: [raceme.foot, mid, raceme.top], scales: [1.0, 0.8, 0.55], capStart: false) {
                peduncles.append(stalk.smoothed())
            }
            let f = frame(yAxis: axis)
            let phase = rng.unit() * 2 * Double.pi
            let head = len3(raceme.top - raceme.foot) * aloeRacemeHead
            let floret = spec.size * 0.032
            for j in 0 ..< aloeFlowersPerRaceme {
                let h = Double(j) / Double(aloeFlowersPerRaceme - 1)
                let az = phase + Double(j) * goldenAngle
                let radial = normalize3(f.x * cos(az) + f.z * sin(az))
                let point = raceme.top - axis * (head * (1 - h)) + radial * (spec.size * 0.006)
                let open = h < aloeOpenBelow
                let dir = open ? normalize3(radial * 0.55 - axis * 0.75) : normalize3(radial * 0.6 + axis * 0.45)
                let length = floret * (open ? 1.0 : 0.75) * (1 - 0.35 * h) * (0.9 + rng.unit() * 0.2)
                let tube = aloeFloret(length: length, radius: length * 0.14, origin: point, axis: dir)
                if open { flowers.append(tube) } else { buds.append(tube) }
            }
        }

        var groups = SpeciesColorGroups()
        groups.append(withered, color: aloeWitheredColor)
        groups.append(blush, color: aloeBlushColor)
        groups.append(wood, color: aloeWoodColor)
        groups.append(peduncles, color: aloePeduncleColor)
        groups.append(flowers, color: aloeFlowerColor)
        groups.append(buds, color: aloeBudColor)
        return Parts(stem: body, foliage: teeth, blooms: groups.groups)
    }
}
