import Foundation
import simd
import VisualizerMaterials

extension LeafSilhouette {
    /// An echeveria leaf: a spoon — narrow at the base, broadening to a rounded paddle two-thirds
    /// out, closing to a short soft point. The mid tier (the fewest controls that still read as a
    /// spoon) is for the cupped inner leaves, the chicks and the stalk's bracts.
    public static let echeveriaLeaf = LeafSilhouette(
        name: "echeveriaLeaf",
        hero: [(0.00, 0.00), (0.12, 0.44), (0.40, 0.78), (0.68, 1.00), (0.86, 0.88), (0.95, 0.44), (1.00, 0.00)],
        mid: [(0.00, 0.00), (0.30, 0.66), (0.78, 1.00), (1.00, 0.00)])
}

/// *Echeveria elegans* — the Mexican snowball, the "hens and chicks" of Southern California's
/// gardens (DH-0797). A small, tight, geometric rosette sitting flush on the soil.
///
/// What makes it read as this plant, trait by trait:
///  - **A tight golden-angle (Fibonacci) rosette** of plump SPOON leaves — solids, on the opaque
///    `stem` part.
///  - **A radial leaf-angle gradient:** the outer leaves splay nearly flat, the inner ones stand up
///    and curl in, cupped.
///  - **Pale, powdery blue-grey**, farina-matte (the opaque part's roughness).
///  - **A base-to-tip colour gradient**: every leaf blushes pink toward its soft point — more of the
///    leaf on the sun-struck outer ring. The blushed tips are the `foliage` part, so the species'
///    `leafColor` is the blush.
///  - **Small** — a rosette ~13 cm across at the default size; no visible stem.
///  - **Chicks on stolons** round the base: many rosette sizes at once, never one lone rosette.
///  - **A wiry flower stalk from the SIDE** of the rosette — out of a leaf axil, never the centre —
///    arching up and nodding, with small fleshy bracts and pendant coral bells.
extension GardenPlantMesh {

    public struct EcheveriaRosette: Sendable, Equatable {
        /// On the ground.
        public var centre: Vec3
        /// The outer leaves' reach.
        public var radius: Double
        public var chick: Bool
    }

    public struct EcheveriaLeaf: Sendable, Equatable {
        public var leaf: FleshyLeaf
        public var rosette: Int
        /// 0 the outermost leaf → 1 the innermost.
        public var rank: Double
        /// The leaf's azimuth round its rosette (radians).
        public var azimuth: Double
        /// The tip past this fraction is blushed.
        public var blushFrom: Double
    }

    public struct EcheveriaStalk: Sendable, Equatable {
        public var path: [Vec3]
        public var bells: [(point: Vec3, axis: Vec3, open: Bool)]
        public var bracts: [FleshyLeaf]

        public static func == (a: EcheveriaStalk, b: EcheveriaStalk) -> Bool {
            a.path == b.path && a.bracts == b.bracts && a.bells.count == b.bells.count
                && zip(a.bells, b.bells).allSatisfy { $0.point == $1.point && $0.axis == $1.axis && $0.open == $1.open }
        }
    }

    public static let echeveriaLeafCount = 34
    public static let echeveriaChickLeafCount = 11
    /// The main rosette's diameter as a fraction of the plant's size.
    public static let echeveriaRosetteDiameter = 0.82
    public static let echeveriaBells = 7
    public static let echeveriaStalkColor = Vec3(0.62, 0.50, 0.44)
    public static let echeveriaBellColor  = Vec3(0.94, 0.46, 0.34)
    public static let echeveriaBudColor   = Vec3(0.84, 0.30, 0.32)

    /// The main rosette first, then the chicks — the single source for the mesh and its tests.
    public static func echeveriaRosettes(spec: GardenPlantSpec) -> [EcheveriaRosette] {
        var rng = speciesRNG(spec, lane: 0xECE0)
        let r = spec.size * echeveriaRosetteDiameter / 2
        var out = [EcheveriaRosette(centre: Vec3(0, 0, 0), radius: r, chick: false)]
        let chicks = max(1, min(5, Int((1 + 2 * Double(spec.foliageDensity)).rounded())))
        let start = rng.unit() * 2 * Double.pi
        for i in 0 ..< chicks {
            let az = start + Double(i) * 2 * Double.pi / Double(chicks) + (rng.unit() - 0.5) * 0.6
            let d = r * (1.30 + rng.unit() * 0.40)
            out.append(EcheveriaRosette(centre: Vec3(d * cos(az), 0, d * sin(az)),
                                        radius: r * (0.28 + rng.unit() * 0.17), chick: true))
        }
        return out
    }

    public static func echeveriaLeaves(spec: GardenPlantSpec, rosettes: [EcheveriaRosette]) -> [EcheveriaLeaf] {
        var rng = speciesRNG(spec, lane: 0xECE1)
        let up = Vec3(0, 1, 0)
        var out: [EcheveriaLeaf] = []
        for (ri, ros) in rosettes.enumerated() {
            let n = ros.chick ? echeveriaChickLeafCount : echeveriaLeafCount
            let phase = rng.unit() * 2 * Double.pi
            let R = ros.radius
            for i in 0 ..< n {
                let t = Double(i) / Double(n - 1)
                let radial = rosetteRadial(axis: up, index: i, phase: phase)
                let pitch = (ros.chick ? 0.30 : 0.10) + 1.15 * pow(t, ros.chick ? 1.2 : 1.6)
                let frame = rosetteLeafFrame(axis: up, radial: radial, pitch: pitch)
                let length = R * (1.02 - 0.62 * t) * (0.96 + rng.unit() * 0.08)
                let width = length * (0.52 + 0.08 * t)
                out.append(EcheveriaLeaf(
                    leaf: FleshyLeaf(base: ros.centre + radial * (R * 0.06 * (1 - t)) + up * (R * (0.05 + 0.22 * t)),
                                     yAxis: frame.yAxis, normal: frame.normal, length: length, width: width,
                                     thickness: width * 0.24, crown: 0.5, tipThickness: 0.55,
                                     channel: 0.03 + 0.10 * t, arch: -(0.12 + 0.30 * t)),
                    rosette: ri, rank: t,
                    azimuth: phase + Double(i) * goldenAngle,
                    blushFrom: 0.70 + 0.16 * t))
            }
        }
        return out
    }

    /// The flower stalk: out of a leaf axil on the main rosette's side, arching up past the rosette
    /// and nodding at the tip, bracts on its lower half and bells on its upper.
    public static func echeveriaStalk(spec: GardenPlantSpec) -> EcheveriaStalk {
        var rng = speciesRNG(spec, lane: 0xECE2)
        let R = spec.size * echeveriaRosetteDiameter / 2
        let up = Vec3(0, 1, 0)
        let az = rng.unit() * 2 * Double.pi
        let out = Vec3(cos(az), 0, sin(az))
        let side = Vec3(-out.z, 0, out.x) * ((rng.unit() - 0.5) * R * 0.3)
        let foot = R * 0.12
        let rise = spec.size * 0.92 - foot
        let stations: [(r: Double, h: Double)] = [(0.30, 0), (0.45, 0.35), (0.62, 0.70), (0.82, 0.93), (1.02, 1.0), (1.18, 0.90)]
        let path = stations.enumerated().map { i, s in
            out * (R * s.r) + up * (foot + rise * s.h) + side * (Double(i) / 5)
        }
        func point(_ tau: Double) -> (Vec3, Vec3) {
            let x = tau * Double(path.count - 1)
            let i = min(path.count - 2, Int(x))
            let f = x - Double(i)
            return (path[i] + (path[i + 1] - path[i]) * f, normalize3(path[i + 1] - path[i]))
        }

        var bracts: [FleshyLeaf] = []
        for (k, tau) in [0.12, 0.26, 0.40].enumerated() {
            let (p, tangent) = point(tau)
            let flank = normalize3(cross3(tangent, up)) * (k % 2 == 0 ? 1 : -1)
            let y = normalize3(tangent * 0.8 + flank * 0.6)
            let n = normalize3(cross3(y, cross3(tangent, y)))   // the face toward the stalk
            let length = R * 0.30 * (1 - 0.25 * Double(k))
            bracts.append(FleshyLeaf(base: p, yAxis: y, normal: len3(n) > 0.5 ? n : frame(yAxis: y).x,
                                     length: length, width: length * 0.45, thickness: length * 0.10,
                                     crown: 0.5, tipThickness: 0.5, channel: 0.05, arch: -0.2))
        }

        var bells: [(point: Vec3, axis: Vec3, open: Bool)] = []
        for k in 0 ..< echeveriaBells {
            let tau = 0.58 + 0.42 * Double(k) / Double(echeveriaBells - 1)
            let (p, tangent) = point(tau)
            let flank = normalize3(cross3(tangent, up)) * (k % 2 == 0 ? 1 : -1)
            let axis = normalize3(up * -0.8 + flank * 0.35 + out * 0.2)
            bells.append((p, axis, k < echeveriaBells - 2))
        }
        return EcheveriaStalk(path: path, bells: bells, bracts: bracts)
    }

    public static func echeveriaParts(spec: GardenPlantSpec) -> Parts {
        let rosettes = echeveriaRosettes(spec: spec)
        var body = Mesh3(), blush = Mesh3()
        for l in echeveriaLeaves(spec: spec, rosettes: rosettes) {
            // The small cupped inner leaves and the chicks are mostly hidden; the splayed outer ring
            // is what the eye reads, so it alone gets the hero outline.
            let tier: LeafSilhouette.Tier = rosettes[l.rosette].chick || l.rank > 0.55 ? .mid : .hero
            let (base, tip) = split(fleshyLeafBuild(l.leaf, silhouette: .echeveriaLeaf, tier: tier), at: l.blushFrom)
            body.append(base)
            blush.append(tip)
        }

        let R = rosettes[0].radius
        var stalks = Mesh3(), bells = Mesh3(), buds = Mesh3()
        for chick in rosettes where chick.chick {
            // A stolon: out from under the main rosette, arching just clear of the soil to the chick.
            let dir = normalize3(Vec3(chick.centre.x, 0, chick.centre.z))
            let a = dir * (R * 0.35) + Vec3(0, R * 0.02, 0)
            let c = chick.centre - dir * (chick.radius * 0.2) + Vec3(0, R * 0.02, 0)
            let b = (a + c) * 0.5 + Vec3(0, R * 0.12, 0)
            var stolon = Mesh3()
            if stolon.sweep(profile: .circle(radius: R * 0.03, segments: 4), along: [a, b, c],
                            capStart: false, capEnd: false) {
                stalks.append(stolon.smoothed())
            }
        }

        let stalk = echeveriaStalk(spec: spec)
        var wire = Mesh3()
        if wire.sweep(profile: .circle(radius: spec.size * 0.012, segments: 5), along: stalk.path,
                      scales: stalk.path.indices.map { 1 - 0.45 * Double($0) / Double(stalk.path.count - 1) },
                      capStart: false) {
            stalks.append(wire.smoothed())
        }
        for bract in stalk.bracts {
            body.append(fleshyLeafBuild(bract, silhouette: .echeveriaLeaf, tier: .mid).mesh)
        }
        for bell in stalk.bells {
            let length = spec.size * (bell.open ? 0.07 : 0.05)
            let r = length * 0.38
            let profile: [(r: Double, y: Double)] = bell.open
                ? [(r * 0.35, 0), (r * 0.8, length * 0.35), (r, length * 0.8), (r * 1.05, length)]
                : [(r * 0.35, 0), (r * 0.75, length * 0.5), (0, length)]
            let mesh = revolvedPart(profile, segments: 6, origin: bell.point, axis: bell.axis)
            if bell.open { bells.append(mesh) } else { buds.append(mesh) }
        }

        var groups = SpeciesColorGroups()
        groups.append(stalks, color: echeveriaStalkColor)
        groups.append(bells, color: echeveriaBellColor)
        groups.append(buds, color: echeveriaBudColor)
        return Parts(stem: body, foliage: blush, blooms: groups.groups)
    }
}
