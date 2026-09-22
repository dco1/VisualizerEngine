import Foundation
import simd
import VisualizerMaterials

/// A cactus flower petal — narrow, spoon-ended, a soft point. Shared by both cacti (DH-0649).
extension LeafSilhouette {
    public static let cactusPetal = LeafSilhouette(
        name: "cactusPetal",
        hero: [(0.00, 0.00), (0.15, 0.40), (0.45, 0.80), (0.75, 0.90), (0.92, 0.56), (1.00, 0.00)])
}

/// *Echinocactus grusonii* — the golden barrel cactus (DH-0649). Not a rosette and not leaves on a
/// stem: a ribbed body wearing golden spines, and nothing else.
///
/// What makes it read as this plant, trait by trait:
///  - **A ribbed sphere that becomes a barrel with age.** `goldenBarrelBody` derives age from
///    `size`: a young plant is wider than tall, an old one taller than wide (a superellipse profile
///    whose exponent rises with age), and the rib count climbs with it.
///  - **Deep, evenly spaced ribs**, a row of areoles down every crest, alternate ribs staggered half
///    a step.
///  - **Golden spines in a starburst** at every areole — four radials lying close to the body round
///    one longer central. They are the `foliage` part (a spine IS a modified leaf), so they take the
///    outdoor thin-sheet backlight: a low sun lights the spine halo, the plant's signature
///    photograph. The species' `leafColor` is therefore gold, not green.
///  - **The body is the opaque, matte `stem` part** (deep green) — a solid, never a backlit sheet.
///  - **A woolly crown** of pale felt tufts in the flattened, sunken apex — and the ONLY place a
///    flower appears.
///  - **No leaves at all.** The foliage part is spines and nothing else (the test counts it).
extension GardenPlantMesh {

    public struct GoldenBarrelBody: Sendable, Equatable {
        /// Body height above the ground (metres); the base sits a little below grade.
        public var height: Double
        /// Radius at the rib crests at the widest point.
        public var radius: Double
        public var ribs: Int
        /// Valley depth as a fraction of the local radius.
        public var ribDepth: Double
        /// Superellipse exponent of the profile: 2 is a sphere, higher squares the shoulders.
        public var squareness: Double
        public var areolesPerRib: Int
        public var spineLength: Double
        /// Flowers in the wool crown (0…3).
        public var flowers: Int
    }

    public static let goldenBarrelSpinesPerAreole = 5
    public static let goldenBarrelWoolColor   = Vec3(0.78, 0.75, 0.66)
    public static let goldenBarrelFlowerColor = Vec3(0.92, 0.72, 0.12)
    /// The profile's lowest point in superellipse units (−1 would be the full sphere's bottom) — the
    /// rest of the body is below grade.
    public static let goldenBarrelBuried = -0.8
    /// Areoles run down each crest between these profile parameters: clear of the soil below, and
    /// stopping where the wool crown takes over above.
    public static let goldenBarrelAreoleBand = (lo: 0.14, hi: 0.86)

    /// The body — the single source the mesh and its tests read.
    public static func goldenBarrelBody(spec: GardenPlantSpec) -> GoldenBarrelBody {
        var rng = speciesRNG(spec, lane: 0x6BA1)
        let h = spec.size
        let age = min(1, max(0, (h - 0.2) / 0.8))   // 0 a 20 cm youngster … 1 a metre-tall veteran
        let ribs = 20 + Int((age * 6).rounded())
        let areoles = 135 * (0.85 + 0.15 * Double(spec.foliageDensity))
        return GoldenBarrelBody(
            height: h,
            radius: h * (0.56 - 0.16 * age) * (0.96 + rng.unit() * 0.08),
            ribs: ribs,
            ribDepth: 0.10 + 0.03 * age,
            squareness: 2.1 + 1.1 * age,
            areolesPerRib: max(5, Int((areoles / Double(ribs)).rounded())),
            spineLength: min(0.055, max(0.018, h * 0.085)),
            flowers: rng.unit() < 0.4 ? 0 : 1 + Int(rng.unit() * 2.99))
    }

    /// The body's crest radius and height at profile parameter `t` (0 = the buried base, 1 = the
    /// apex), sampled by ANGLE round a superellipse so the rounded shoulders get their share of rings.
    public static func goldenBarrelProfile(_ b: GoldenBarrelBody, t: Double) -> (r: Double, y: Double) {
        let e = 2 / b.squareness
        let phiBase = acos(pow(-goldenBarrelBuried, b.squareness / 2))
        let phi = phiBase + (Double.pi - phiBase) * t
        let s = sin(phi), c = cos(phi)
        let r = b.radius * pow(max(0, s), e)
        let yn = (c > 0 ? -1.0 : 1.0) * pow(abs(c), e)
        var y = (yn - goldenBarrelBuried) / (1 - goldenBarrelBuried) * b.height * 1.03 - b.height * 0.03
        y -= b.height * 0.07 * (1 - smoothstep(0, 0.4, r / b.radius))   // the sunken woolly apex
        return (r, y)
    }

    /// 0 on a rib crest → 1 in a valley. Sharp at the crest, rounded in the valley.
    public static func goldenBarrelValley(_ theta: Double, ribs: Int) -> Double {
        pow(max(0, 0.5 - 0.5 * cos(Double(ribs) * theta)), 0.65)
    }

    /// Surface height at plan radius `r` in the crown (the apex side of the profile).
    public static func goldenBarrelCrownY(_ b: GoldenBarrelBody, radius r: Double) -> Double {
        var t = 1.0
        while t > 0.5 {
            let p = goldenBarrelProfile(b, t: t)
            if p.r >= r { return p.y }
            t -= 0.005
        }
        return goldenBarrelProfile(b, t: 0.5).y
    }

    public static func goldenBarrelBodyMesh(_ b: GoldenBarrelBody) -> Mesh3 {
        let perRib = 6, rings = 12
        let around = b.ribs * perRib
        let centre = Vec3(0, b.height * 0.45, 0)
        var grid: [[Vec3]] = []
        for i in 0 ..< rings {
            let (r, y) = goldenBarrelProfile(b, t: Double(i) / Double(rings))
            grid.append((0 ..< around).map { j in
                let th = 2 * Double.pi * Double(j) / Double(around)
                let rr = r * (1 - b.ribDepth * goldenBarrelValley(th, ribs: b.ribs))
                return Vec3(rr * cos(th), y, rr * sin(th))
            })
        }
        let apex = Vec3(0, goldenBarrelProfile(b, t: 1).y, 0)
        let floor = Vec3(0, grid[0][0].y, 0)
        var m = Mesh3()
        for i in 0 ..< rings - 1 {
            for j in 0 ..< around {
                let k = (j + 1) % around
                let a = grid[i][j], bb = grid[i][k], c = grid[i + 1][k], d = grid[i + 1][j]
                m.addQuad(a, bb, c, d, outward: (a + bb + c + d) * 0.25 - centre)
            }
        }
        let top = grid[rings - 1]
        for j in 0 ..< around {
            let k = (j + 1) % around
            m.addTriangle(top[j], top[k], apex, outward: (top[j] + top[k] + apex) * (1.0 / 3.0) - centre)
            m.addTriangle(grid[0][j], grid[0][k], floor, outward: Vec3(0, -1, 0))
        }
        return m.smoothed(creaseDegrees: 50)
    }

    /// A slender three-sided needle from `base` to `tip`. Left flat: at a millimetre across, the
    /// facets are sub-pixel, and smoothing would only blur the glint down its length.
    public static func appendNeedle(_ m: inout Mesh3, base: Vec3, tip: Vec3, radius: Double) {
        guard len3(tip - base) > 1e-6 else { return }
        let f = frame(yAxis: tip - base)
        let ring = (0 ..< 3).map { i -> Vec3 in
            let a = Double(i) * 2 * Double.pi / 3
            return base + (f.x * cos(a) + f.z * sin(a)) * radius
        }
        for i in 0 ..< 3 {
            let p = ring[i], q = ring[(i + 1) % 3]
            m.addTriangle(p, q, tip, outward: (p + q) * 0.5 - base)
        }
    }

    public static func goldenBarrelSpines(_ b: GoldenBarrelBody, rng: inout PottedPlantMesh.SplitMix) -> Mesh3 {
        var m = Mesh3()
        let band = goldenBarrelAreoleBand
        for rib in 0 ..< b.ribs {
            let th = 2 * Double.pi * Double(rib) / Double(b.ribs)
            let radial = Vec3(cos(th), 0, sin(th))
            let around = Vec3(-sin(th), 0, cos(th))
            for k in 0 ..< b.areolesPerRib {
                let t = band.lo + (band.hi - band.lo)
                    * (Double(k) + (rib % 2 == 0 ? 0.25 : 0.75)) / Double(b.areolesPerRib)
                let (r, y) = goldenBarrelProfile(b, t: t)
                let lo = goldenBarrelProfile(b, t: t - 0.01), hi = goldenBarrelProfile(b, t: t + 0.01)
                let n = normalize3(radial * (hi.y - lo.y) + Vec3(0, lo.r - hi.r, 0))
                let along = normalize3(cross3(around, n))
                let base = radial * r + Vec3(0, y, 0) + n * (b.spineLength * 0.02)
                for s in 0 ..< goldenBarrelSpinesPerAreole - 1 {
                    let a = Double(s) * Double.pi / 2 + Double.pi / 4 + (rng.unit() - 0.5) * 0.5
                    let dir = normalize3(n * 0.45 + around * cos(a) + along * sin(a))
                    let len = b.spineLength * (0.75 + rng.unit() * 0.3)
                    appendNeedle(&m, base: base, tip: base + dir * len, radius: max(0.0007, len * 0.03))
                }
                let central = normalize3(n + along * 0.35 + around * ((rng.unit() - 0.5) * 0.3))
                let len = b.spineLength * 1.3
                appendNeedle(&m, base: base, tip: base + central * len, radius: max(0.0008, len * 0.03))
            }
        }
        return m
    }

    public static func goldenBarrelParts(spec: GardenPlantSpec) -> Parts {
        let b = goldenBarrelBody(spec: spec)
        var rng = speciesRNG(spec, lane: 0x6BA2)
        let body = goldenBarrelBodyMesh(b)
        let spines = goldenBarrelSpines(b, rng: &rng)
        var groups = SpeciesColorGroups()
        let up = Vec3(0, 1, 0)

        // The wool crown: pale felt tufts packed into the sunken apex.
        let woolR = b.radius * 0.30
        var wool = Mesh3()
        let blobs = 8
        let start = rng.unit() * 2 * Double.pi
        for k in 0 ... blobs {
            let rad = k == 0 ? 0 : woolR * (0.45 + rng.unit() * 0.35)
            let ang = start + Double(k) * 2 * Double.pi / Double(blobs)
            let rb = b.radius * (0.085 + rng.unit() * 0.04)
            let origin = Vec3(rad * cos(ang), goldenBarrelCrownY(b, radius: rad) - rb * 0.35, rad * sin(ang))
            wool.append(revolvedPart([(rb, 0), (rb * 0.92, rb * 0.45), (rb * 0.6, rb * 0.8), (0, rb * 0.92)],
                                     segments: 7, origin: origin, axis: up))
        }
        groups.append(wool, color: goldenBarrelWoolColor)

        // Flowers, only ever in the crown.
        let petalLen = min(0.03, max(0.014, b.height * 0.05))
        for _ in 0 ..< b.flowers {
            let ang = rng.unit() * 2 * Double.pi
            let rad = woolR * 0.75
            let out = Vec3(cos(ang), 0, sin(ang))
            let center = out * rad + Vec3(0, goldenBarrelCrownY(b, radius: rad) + b.radius * 0.05, 0)
            let axis = normalize3(up + out * 0.5)
            let f = frame(yAxis: axis)
            for k in 0 ..< 8 {
                let a = Double(k) * Double.pi / 4 + (rng.unit() - 0.5) * 0.2
                let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                emitBandedPetal(into: &groups,
                                placement: petalPlacement(center: center, axis: axis, radial: radial,
                                                          pitch: 0.62 + (k % 2 == 0 ? 0 : 0.12),
                                                          length: petalLen, width: petalLen * 0.48),
                                silhouette: .cactusPetal, subdivisions: 1, fold: 0.2, curl: -0.08,
                                bands: [PetalBand(vEnd: 1, color: goldenBarrelFlowerColor)])
            }
        }
        return Parts(stem: body, foliage: spines, blooms: groups.groups)
    }
}
