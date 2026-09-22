import Foundation
import simd
import VisualizerMaterials

/// *Strelitzia reginae* — bird of paradise (DH-0796). An ARCHITECTURAL CLUMP of leaf fans.
///
/// What makes it read as this plant, trait by trait:
///  - **Paddle leaves on long rigid petioles, in FANS.** Every leaf of a fan leaves the crown in one
///    vertical plane, splayed like a hand of cards; several fans share the base (`birdOfParadiseLeaves`).
///  - **Wind-torn blades**: most leaves are split from the margin to the midrib at a few places
///    (`StrelitziaLeaf.tears`) — the pieces stay joined only by the midrib, as on a real plant.
///  - **A prominent pale midrib** down every blade, and a glossy deep blue-green surface.
///  - **The crane's head on a BARE stalk** that rises from the base and tops the leaf clump
///    (`birdOfParadiseFlowers`): a hard, horizontal, beak-shaped spathe, green below and streaked
///    maroon along its upper lip, with ONE or TWO flowers out of it at a time — each a crest of three
///    orange sepals and a blue arrow-shaped tongue.
extension GardenPlantMesh {

    public struct StrelitziaLeaf: Sendable, Equatable {
        public var fan: Int
        public var foot: Vec3
        /// The petiole's (straight, rigid) direction; the blade continues along it.
        public var direction: Vec3
        public var petioleLength: Double
        public var bladeLength: Double
        public var bladeWidth: Double
        /// The blade's presented face (perpendicular to `direction`); its droop bows away from it.
        public var face: Vec3
        /// Where the blade is torn through to the midrib, as fractions of its length.
        public var tears: [Double]

        public var petioleTop: Vec3 { foot + direction * petioleLength }
    }

    public struct StrelitziaFlower: Sendable, Equatable {
        public var foot: Vec3
        /// The top of the bare stalk, where the spathe sits.
        public var neck: Vec3
        /// The horizontal direction the spathe's beak points.
        public var beak: Vec3
        public var spatheLength: Double
        /// Flowers out of the spathe at once — never more than two.
        public var openFlorets: Int
    }

    public static let strelitziaBladeCurl = 0.10
    /// The width of a tear, as a fraction of the blade length.
    public static let strelitziaTearGap = 0.014

    public static let strelitziaMidrib = Vec3(0.62, 0.66, 0.50)
    public static let strelitziaSpathe = Vec3(0.30, 0.40, 0.26)
    public static let strelitziaMaroon = Vec3(0.32, 0.09, 0.11)
    public static let strelitziaOrange = Vec3(0.95, 0.38, 0.03)
    public static let strelitziaBlue   = Vec3(0.22, 0.28, 0.80)

    /// The leaves — the single source the mesh and the tests read.
    public static func birdOfParadiseLeaves(spec: GardenPlantSpec) -> [StrelitziaLeaf] {
        var rng = speciesRNG(spec, lane: 0x5781)
        let size = spec.size, up = Vec3(0, 1, 0)
        let fans = max(2, min(4, Int((2.5 * Double(spec.foliageDensity)).rounded())))
        let perFan = max(3, min(6, Int((4.5 * Double(spec.foliageDensity)).rounded())))
        let start = rng.unit() * Double.pi
        var leaves: [StrelitziaLeaf] = []
        for fi in 0 ..< fans {
            let theta = start + Double(fi) * Double.pi / Double(fans) + (rng.unit() - 0.5) * 0.3
            let fanAxis = Vec3(cos(theta), 0, sin(theta)), fanNormal = Vec3(-sin(theta), 0, cos(theta))
            // Fans stand round the crown, not on top of one another.
            let centre = fanNormal * (size * 0.05 * (Double(fi) - Double(fans - 1) / 2))
            let spread = 0.38 + rng.unit() * 0.16
            for k in 0 ..< perFan {
                let a = -spread + 2 * spread * Double(k) / Double(perFan - 1) + (rng.unit() - 0.5) * 0.10
                let direction = normalize3(up * cos(a) + fanAxis * sin(a))
                var face = up - direction * dot3(up, direction)
                face = len3(face) > 1e-6 ? normalize3(face) * (abs(sin(a)) + 0.25) : Vec3(0, 0, 0)
                face = face + fanNormal * ((rng.unit() - 0.5) * 1.2)
                face = face - direction * dot3(face, direction)
                face = len3(face) > 1e-6 ? normalize3(face) : fanNormal
                let bladeLength = size * (0.24 + rng.unit() * 0.06)
                var tears: [Double] = []
                if rng.unit() < 0.75 {
                    let n = 1 + Int(rng.unit() * 3)
                    tears = (0 ..< n).map { _ in 0.25 + rng.unit() * 0.65 }.sorted()
                }
                leaves.append(StrelitziaLeaf(
                    fan: fi, foot: centre + fanAxis * (size * 0.02 * (rng.unit() - 0.5)),
                    direction: direction,
                    petioleLength: size * (0.40 + rng.unit() * 0.08) * (1 - 0.15 * abs(a) / spread),
                    bladeLength: bladeLength, bladeWidth: bladeLength * 0.36, face: face, tears: tears))
            }
        }
        // A real clump always shows weathering somewhere.
        if !leaves.isEmpty && !leaves.contains(where: { !$0.tears.isEmpty }) { leaves[0].tears = [0.55] }
        return leaves
    }

    /// The flower stalks — each topping the whole leaf clump.
    public static func birdOfParadiseFlowers(spec: GardenPlantSpec) -> [StrelitziaFlower] {
        var rng = speciesRNG(spec, lane: 0x5782)
        let size = spec.size, up = Vec3(0, 1, 0)
        let count = max(1, min(3, Int((1.5 * Double(spec.foliageDensity)).rounded())))
        return (0 ..< count).map { i in
            let az = rng.unit() * 2 * Double.pi
            let out = Vec3(cos(az), 0, sin(az))
            let foot = out * (size * 0.03 * rng.unit())
            let beakAz = az + (rng.unit() - 0.5) * 1.6
            return StrelitziaFlower(
                foot: foot,
                neck: foot + out * (size * (0.05 + rng.unit() * 0.10)) + up * (size * (0.83 + rng.unit() * 0.07)),
                beak: Vec3(cos(beakAz), 0, sin(beakAz)),
                spatheLength: size * (0.17 + rng.unit() * 0.03),
                openFlorets: i % 2 == 0 ? 1 : 2)
        }
    }

    public static func birdOfParadiseParts(spec: GardenPlantSpec) -> Parts {
        let size = spec.size
        var stalks = Mesh3(), foliage = Mesh3(), midribs = Mesh3()
        var groups = SpeciesColorGroups()
        var rng = speciesRNG(spec, lane: 0x5783)

        for leaf in birdOfParadiseLeaves(spec: spec) {
            PottedPlantMesh.appendBentStemTube(&stalks, from: leaf.foot, to: leaf.petioleTop,
                                               rBase: size * 0.0075, rTip: size * 0.0050,
                                               bendDir: leaf.face * -1, bendAmount: 0.02, sides: 5)
            foliage.append(strelitziaBlade(leaf))

            // The pale midrib, following the blade's spine and droop — and holding the torn pieces.
            let h = leaf.bladeLength, curl = strelitziaBladeCurl
            let spine = [0.0, 0.33, 0.66, 0.95].map { v in
                leaf.petioleTop + leaf.direction * (h * v) + leaf.face * (-curl * h * v * v)
            }
            var rib = Mesh3()
            if rib.sweep(profile: .circle(radius: leaf.bladeWidth * 0.028, segments: 4), along: spine,
                         scales: [1, 0.85, 0.6, 0.2]) {
                midribs.append(rib.smoothed())
            }
        }
        groups.append(midribs, color: strelitziaMidrib)

        for flower in birdOfParadiseFlowers(spec: spec) {
            let beak = flower.beak
            let (tip, _) = PottedPlantMesh.appendBentStemTube(&stalks, from: flower.foot, to: flower.neck,
                                                              rBase: size * 0.0085, rTip: size * 0.0060,
                                                              bendDir: beak, bendAmount: 0.04, sides: 5)
            emitStrelitziaHead(into: &groups, neck: tip, beak: beak, length: flower.spatheLength,
                               openFlorets: flower.openFlorets, rng: &rng)
        }

        return Parts(stem: stalks, foliage: foliage, blooms: groups.groups)
    }

    /// One paddle blade, in pieces between its tears. The pieces share ONE placement, so the fold
    /// and droop match across every tear, and are smoothed as one part. The blade runs along the
    /// petiole's direction from its top, so a vertex's fraction along the blade is exactly
    /// `dot(p − petioleTop, direction) / bladeLength` (fold and droop are both perpendicular to it).
    public static func strelitziaBlade(_ leaf: StrelitziaLeaf) -> Mesh3 {
        let h = leaf.bladeLength
        let placement = LeafConstructor.Placement<Double>(
            position: leaf.petioleTop + leaf.direction * (h * 0.46),
            xAxis: normalize3(cross3(leaf.direction, leaf.face)), yAxis: leaf.direction,
            bentNormal: leaf.face, width: leaf.bladeWidth, height: h)
        let paddle = LeafSilhouette.strelitziaPaddle.margin(tier: .hero, subdivisions: 2)
        var cuts: [(Double, Double)] = []
        var v0 = 0.0
        for t in leaf.tears {
            cuts.append((v0, t - strelitziaTearGap / 2))
            v0 = t + strelitziaTearGap / 2
        }
        cuts.append((v0, 1.0))
        var blade = Mesh3()
        for (a, b) in cuts {
            let slice = marginSlice(paddle, from: a, to: b)
            if slice.count >= 2 {
                LeafConstructor.emitBlade(into: &blade, placement: placement, margin: slice,
                                          fold: 0.18, curl: strelitziaBladeCurl, winding: .doubleSidedShell)
            }
        }
        return blade.smoothed()
    }

    /// The spathe's centre-line stations: fraction along the beak, and the section scale there.
    public static let strelitziaSpatheStations: [(s: Double, scale: Double)] =
        [(0.00, 0.30), (0.16, 0.85), (0.34, 1.00), (0.52, 0.92), (0.70, 0.70), (0.86, 0.40), (1.00, 0.06)]

    /// The crane's head: the spathe (green, maroon along its upper lip) and its flowers.
    private static func emitStrelitziaHead(into groups: inout SpeciesColorGroups, neck: Vec3, beak: Vec3,
                                           length L: Double, openFlorets: Int,
                                           rng: inout PottedPlantMesh.SplitMix) {
        let up = Vec3(0, 1, 0)
        let r = L * 0.10
        let lateral = normalize3(cross3(beak, up))
        func centre(_ s: Double) -> Vec3 { neck + beak * (L * (s - 0.2)) + up * (L * 0.05 * sin(s * Double.pi)) }
        func scale(at s: Double) -> Double {
            let st = strelitziaSpatheStations
            for i in 0 ..< st.count - 1 where s <= st[i + 1].s {
                let f = (s - st[i].s) / (st[i + 1].s - st[i].s)
                return st[i].scale + (st[i + 1].scale - st[i].scale) * max(0, min(1, f))
            }
            return st[st.count - 1].scale
        }

        // A laterally compressed section: tall, narrow — a boat's hull, not a round pod.
        let section = Mesh3.SectionProfile(points: (0 ..< 8).map { k in
            let a = Double(k) * 2 * Double.pi / 8
            return Vec2(r * cos(a), r * 0.62 * sin(a))
        })
        var spathe = Mesh3()
        guard spathe.sweep(profile: section, along: strelitziaSpatheStations.map { centre($0.s) },
                           scales: strelitziaSpatheStations.map(\.scale), up: up) else { return }
        spathe = spathe.smoothed()
        var green: [Int] = [], maroon: [Int] = []
        for t in 0 ..< spathe.triangleCount {
            let c = (spathe.positions[Int(spathe.indices[t * 3])] + spathe.positions[Int(spathe.indices[t * 3 + 1])]
                     + spathe.positions[Int(spathe.indices[t * 3 + 2])]) / 3
            let s = dot3(c - neck, beak) / L + 0.2
            let rise = c.y - centre(max(0, min(1, s))).y
            if rise > 0.55 * r * scale(at: s) { maroon.append(t) } else { green.append(t) }
        }
        groups.append(triangles(of: spathe, indices: green), color: strelitziaSpathe)
        groups.append(triangles(of: spathe, indices: maroon), color: strelitziaMaroon)

        // One or two flowers out of the upper seam, the second younger and only half emerged.
        for n in 0 ..< openFlorets {
            let s = n == 0 ? 0.42 : 0.30
            let grown = n == 0 ? 1.0 : 0.6
            let at = centre(s) + up * (r * scale(at: s) * 0.8)
            for k in 0 ..< 3 {
                let y = normalize3(up * 0.9 + beak * 0.35 + lateral * (Double(k - 1) * 0.22 + (rng.unit() - 0.5) * 0.06))
                emitStrelitziaPart(into: &groups, at: at, yAxis: y, sideways: lateral,
                                   length: L * (0.80 + rng.unit() * 0.12) * grown, aspect: 0.30,
                                   color: strelitziaOrange)
            }
            emitStrelitziaPart(into: &groups, at: at + beak * (L * 0.02), yAxis: normalize3(beak * 0.75 + up * 0.65),
                               sideways: lateral, length: L * 0.55 * grown, aspect: 0.13, color: strelitziaBlue)
        }
    }

    private static func emitStrelitziaPart(into groups: inout SpeciesColorGroups, at: Vec3, yAxis: Vec3,
                                           sideways: Vec3, length: Double, aspect: Double, color: Vec3) {
        var face = sideways - yAxis * dot3(sideways, yAxis)
        face = len3(face) > 1e-6 ? normalize3(face) : Vec3(0, 1, 0)
        emitBandedPetal(into: &groups,
                        placement: .init(position: at + yAxis * (length * 0.46),
                                         xAxis: normalize3(cross3(yAxis, face)), yAxis: yAxis,
                                         bentNormal: face, width: length * aspect, height: length),
                        silhouette: .strelitziaSepal, subdivisions: 1, fold: 0.35, curl: 0.05,
                        bands: [PetalBand(vEnd: 1.0, color: color)])
    }

    /// A copy of the listed triangles of `mesh` (vertices re-indexed).
    public static func triangles(of mesh: Mesh3, indices list: [Int]) -> Mesh3 {
        var positions: [Vec3] = [], normals: [Vec3] = [], uvs: [Vec2] = [], indices: [UInt32] = []
        let hasUV = mesh.uvs.count == mesh.positions.count
        for t in list {
            for k in 0 ..< 3 {
                let i = Int(mesh.indices[t * 3 + k])
                indices.append(UInt32(positions.count))
                positions.append(mesh.positions[i])
                normals.append(mesh.normals[i])
                uvs.append(hasUV ? mesh.uvs[i] : .zero)
            }
        }
        return Mesh3(positions: positions, normals: normals, uvs: uvs, indices: indices)
    }
}

extension LeafSilhouette {
    /// A strelitzia paddle: a narrow base where the petiole joins, long rounded shoulders, a blunt
    /// tapering point — banana-like, never a strap.
    public static let strelitziaPaddle = LeafSilhouette(
        name: "strelitziaPaddle",
        hero: [(0.00, 0.00), (0.08, 0.52), (0.30, 0.94), (0.62, 1.00), (0.86, 0.78), (1.00, 0.00)])

    /// A strelitzia sepal (and the blue tongue): a long narrow lance.
    public static let strelitziaSepal = LeafSilhouette(
        name: "strelitziaSepal",
        hero: [(0.00, 0.00), (0.22, 1.00), (0.70, 0.62), (1.00, 0.00)])
}
