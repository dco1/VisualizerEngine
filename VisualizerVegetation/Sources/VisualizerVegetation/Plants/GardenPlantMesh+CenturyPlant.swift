import Foundation
import simd
import VisualizerMaterials

extension LeafSilhouette {
    /// An agave leaf: clasps the core narrow, is widest a third of the way out, then runs a long
    /// straight taper into the terminal spine. The mid tier is the pups'.
    public static let agaveLeaf = LeafSilhouette(
        name: "agaveLeaf",
        hero: [(0.00, 0.00), (0.05, 0.64), (0.18, 0.92), (0.36, 1.00), (0.58, 0.80), (0.80, 0.44), (1.00, 0.00)],
        mid: [(0.00, 0.00), (0.08, 0.72), (0.36, 1.00), (0.74, 0.50), (1.00, 0.00)])
}

/// *Agave americana* — the century plant (DH-0797). A rosette of huge, rigid, thick leaves, and
/// armour.
///
/// What makes it read as this plant, trait by trait:
///  - **Scale.** Leaves over a metre long at the default size, radiating to a rosette well over two
///    metres across (`spreadPerHeight` carries the spread past the 1.5 m `size` clamp).
///  - **Rigid, thick, channelled leaves**, symmetric round the centre on the golden angle — the
///    outer ones splayed low and recurved, the inner ones rising into the tight central spike.
///  - **Hooked marginal teeth** down both margins and a **long terminal spine** on every leaf — the
///    `foliage` part is exactly this armour (the species' `leafColor` is its dark brown).
///  - **Bud imprints:** a pale ghost of the neighbour's toothed margin, pressed into the upper face
///    while the leaves were folded together in the spike, mirrored either side of the midrib.
///  - **Chalky blue-grey** leaves on the opaque `stem` part.
///  - **Pre-bloom.** A century plant flowers once, at the end of its life; a garden specimen is the
///    rosette, so there is no flower stalk.
///  - **The outer leaves are damaged** — tips died back to straw, now and then one broken and
///    folded over — while the inner leaves stay pristine.
///  - **No visible stem**: the leaves erupt straight from a buried core.
///  - **Pups ring the base**, small rosettes of their own.
extension GardenPlantMesh {

    public struct CenturyPlantLeaf: Sendable, Equatable {
        public var leaf: FleshyLeaf
        /// 0 the outermost (oldest) leaf of its rosette → 1 the innermost.
        public var rank: Double
        /// The fraction of the leaf, from the tip, that has died back to straw (0 = pristine).
        public var dieBack: Double
        /// Carries the bud imprint of a neighbour's margin.
        public var imprinted: Bool
        /// Belongs to an offset pup, not the mother rosette.
        public var pup: Bool
    }

    public static let centuryPlantDeadTip = Vec3(0.55, 0.45, 0.30)
    public static let centuryPlantImprint = Vec3(0.62, 0.68, 0.66)
    public static let centuryPlantTeethPerSide = 10
    public static let centuryPlantPupLeaves = 7
    /// A tooth's length and the terminal spine's, as fractions of the leaf's length.
    public static let centuryPlantToothFraction = 0.010
    public static let centuryPlantSpineFraction = 0.032

    public static func centuryPlantLeafCount(_ spec: GardenPlantSpec) -> Int {
        min(28, 20 + Int(4 * Double(spec.foliageDensity)))
    }

    /// The pups: rosette centres on the ground round the mother, and each one's scale.
    public static func centuryPlantPups(spec: GardenPlantSpec) -> [(centre: Vec3, scale: Double)] {
        var rng = speciesRNG(spec, lane: 0xA6A1)
        let count = max(1, min(5, Int(1 + 2 * Double(spec.foliageDensity))))
        let start = rng.unit() * 2 * Double.pi
        return (0 ..< count).map { i in
            let az = start + Double(i) * 2 * Double.pi / Double(count) + (rng.unit() - 0.5) * 0.8
            let r = spec.size * (0.62 + rng.unit() * 0.30)
            return (Vec3(r * cos(az), 0, r * sin(az)), 0.16 + rng.unit() * 0.14)
        }
    }

    /// Every leaf, the mother's (outermost first) then the pups' — the single source the mesh and
    /// its tests read.
    public static func centuryPlantLeaves(spec: GardenPlantSpec) -> [CenturyPlantLeaf] {
        var rng = speciesRNG(spec, lane: 0xA6A0)
        let up = Vec3(0, 1, 0)
        var out: [CenturyPlantLeaf] = []

        func rosette(centre: Vec3, scale: Double, count: Int, pup: Bool) {
            let size = spec.size * scale
            let phase = rng.unit() * 2 * Double.pi
            for i in 0 ..< count {
                let t = Double(i) / Double(count - 1)
                let radial = rosetteRadial(axis: up, index: i, phase: phase)
                let pitch = 0.28 + 1.14 * pow(t, 0.85) + (rng.unit() - 0.5) * 0.08
                let frame = rosetteLeafFrame(axis: up, radial: radial, pitch: pitch)
                let length = size * (1.20 - 0.45 * t) * (0.94 + rng.unit() * 0.12)
                let width = length * (0.17 - 0.04 * t)
                out.append(CenturyPlantLeaf(
                    leaf: FleshyLeaf(base: centre + radial * (size * 0.035 * (1 - t)) + up * (size * (0.015 + 0.09 * t)),
                                     yAxis: frame.yAxis, normal: frame.normal, length: length, width: width,
                                     thickness: width * 0.12, crown: 0.6, tipThickness: 0.25, channel: 0.10,
                                     arch: 0.04 + 0.30 * (1 - t) + (rng.unit() - 0.5) * 0.06),
                    rank: t, dieBack: 0, imprinted: !pup && t >= 0.2, pup: pup))
            }
        }

        let mother = centuryPlantLeafCount(spec)
        rosette(centre: Vec3(0, 0, 0), scale: 1, count: mother, pup: false)
        // The outer third is old and weathered: tips died back, and now and then one leaf broken and
        // folded over. The outermost always shows it; the inner leaves never do.
        for i in 0 ..< mother where out[i].rank < 0.34 {
            if i == 0 || rng.unit() < 0.55 { out[i].dieBack = 0.08 + rng.unit() * 0.18 }
        }
        if rng.unit() < 0.6 {
            let broken = Int(rng.unit() * 4) % mother
            out[broken].leaf.arch = 1.25
            out[broken].dieBack = max(out[broken].dieBack, 0.12)
        }
        for pup in centuryPlantPups(spec: spec) {
            rosette(centre: pup.centre, scale: pup.scale, count: centuryPlantPupLeaves, pup: true)
        }
        return out
    }

    /// The bud imprint: the ghost of the neighbour's toothed margin, a pale zig-zag strip lifted a
    /// hair off the upper face, mirrored either side of the midrib.
    public static func appendBudImprint(_ m: inout Mesh3, leaf l: FleshyLeaf, margin: [LeafSilhouette.Control]) {
        let segments = 6
        let lift = l.thickness * 0.10
        let du = 0.025
        for side in [1.0, -1.0] {
            var previous: (a: Vec3, b: Vec3)?
            for s in 0 ... segments {
                let v = 0.22 + 0.50 * Double(s) / Double(segments)
                let u = 0.62 + (s % 2 == 0 ? -0.07 : 0.07)
                let a = fleshyLeafUpperFace(l, margin: margin, v: v, u: u - du, side: side, lift: lift)
                let b = fleshyLeafUpperFace(l, margin: margin, v: v, u: u + du, side: side, lift: lift)
                if let p = previous {
                    let outward = fleshyLeafUpperFace(l, margin: margin, v: v, u: u, side: side, lift: lift + l.thickness)
                        - fleshyLeafUpperFace(l, margin: margin, v: v, u: u, side: side, lift: lift)
                    m.addQuad(p.a, p.b, b, a, outward: outward)
                }
                previous = (a, b)
            }
        }
    }

    public static func centuryPlantParts(spec: GardenPlantSpec) -> Parts {
        let hero = LeafSilhouette.agaveLeaf.margin(tier: .hero)
        let mid = LeafSilhouette.agaveLeaf.margin(tier: .mid)
        var body = Mesh3(), armour = Mesh3(), dead = Mesh3(), imprint = Mesh3()
        for l in centuryPlantLeaves(spec: spec) {
            let margin = l.pup ? mid : hero
            let built = fleshyLeafBuild(l.leaf, silhouette: .agaveLeaf, tier: l.pup ? .mid : .hero)
            let (live, tip) = split(built, at: 1 - l.dieBack)
            body.append(live)
            dead.append(tip)
            if !l.pup {
                appendMarginTeeth(&armour, leaf: l.leaf, margin: margin, perSide: centuryPlantTeethPerSide,
                                  from: 0.10, to: 0.92, length: l.leaf.length * centuryPlantToothFraction,
                                  hook: 0.7, stoutness: 0.35)
            }
            let spine = fleshyLeafTip(l.leaf)
            let spineLength = l.leaf.length * centuryPlantSpineFraction
            let foot = spine.point - spine.direction * (spineLength * 0.2)
            appendNeedle(&armour, base: foot, tip: foot + spine.direction * spineLength,
                         radius: max(0.0008, spineLength * 0.07))
            if l.imprinted { appendBudImprint(&imprint, leaf: l.leaf, margin: margin) }
        }
        // The buried core the leaves erupt from.
        let r = spec.size * 0.08
        body.append(revolvedPart([(r, -spec.size * 0.03), (r * 0.95, spec.size * 0.05), (r * 0.5, spec.size * 0.10), (0, spec.size * 0.12)],
                                 segments: 10, origin: Vec3(0, 0, 0), axis: Vec3(0, 1, 0)))
        var groups = SpeciesColorGroups()
        groups.append(dead, color: centuryPlantDeadTip)
        groups.append(imprint, color: centuryPlantImprint)
        return Parts(stem: body, foliage: armour, blooms: groups.groups)
    }
}
