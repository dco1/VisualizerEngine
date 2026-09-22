import Foundation
import simd
import VisualizerMaterials

extension LeafSilhouette {
    /// A bunny-ears pad: near-round oval, pinched to a constricted joint at the base.
    public static let bunnyEarPad = LeafSilhouette(
        name: "bunnyEarPad",
        hero: [(0.00, 0.00), (0.06, 0.36), (0.20, 0.74), (0.45, 0.98), (0.70, 0.94), (0.88, 0.68), (1.00, 0.00)])
}

/// *Opuntia microdasys* 'Monstrosa' — the monstrose bunny ears cactus (DH-0649). Not a rosette and
/// not leaves on a stem: a branching chain of flat fleshy PADS.
///
/// What makes it read as this plant, trait by trait:
///  - **Pads, each a closed fleshy solid** (`PadConstructor` — a pancake with a crowned middle and a
///    rounded rim you can see the thickness of), branching from each other at odd angles through a
///    visibly constricted joint, from an older pad's EDGE or, now and then, its flat FACE.
///  - **Monstrose: no two pads share a shape.** Every pad draws its own aspect, cup, bow, outline
///    wobble and two lumps (`BunnyEarPad`), so the silhouette is asymmetric by construction.
///  - **No spines — a polka-dot lattice of golden glochid tufts** on both faces, in a regular
///    staggered grid (`bunnyEarsTufts`). The tufts are the `foliage` part (an areole's glochids are
///    its leaves), so the species' `leafColor` is their tan.
///  - **Pale blue-green matte pads** — the opaque `stem` part. The oldest pads at the base are
///    wrinkled; the young ones at the tips stay plump and smooth.
///  - **A few yellow flowers** sitting straight on the rim of the youngest pads, no stalk.
extension GardenPlantMesh {

    public struct BunnyEarPad: Sendable, Equatable {
        public var parent: Int?
        public var depth: Int
        /// The pad's constricted base — on its parent, or just below grade.
        public var joint: Vec3
        /// Joint → tip.
        public var yAxis: Vec3
        /// The front face.
        public var normal: Vec3
        public var height: Double
        public var width: Double
        /// Rim thickness (the centre is crowned thicker by `PadConstructor.defaultCrown`).
        public var thickness: Double
        // The monstrose deformity — drawn per pad.
        public var cup: Double
        public var bow: Double
        public var lobe1: Double, lobe1Phase: Double
        public var lobe2: Double, lobe2Phase: Double
        /// Two lumps: (across, along) in pad units −1…1, amplitude × thickness, face ±1.
        public var lumps: [SIMD4<Double>]
        /// 0 plump and young → 1 old and wrinkled.
        public var wrinkle: Double

        public var xAxis: Vec3 { normalize3(cross3(yAxis, normal)) }
    }

    public struct BunnyEarsFlower: Sendable, Equatable {
        public var pad: Int
        public var point: Vec3
        public var axis: Vec3
    }

    public static let bunnyEarsFlowerColor  = Vec3(0.96, 0.84, 0.22)
    public static let bunnyEarsThroatColor  = Vec3(0.88, 0.60, 0.16)
    public static let bunnyEarsStamenColor  = Vec3(0.93, 0.90, 0.60)
    /// A glochid tuft's radius as a fraction of its pad's height — a dot, never a spine.
    public static let bunnyEarsTuftRadius = 0.022

    /// The pad's half-width fraction at `v` (0 joint → 1 tip), off the same outline the mesh uses.
    public static func bunnyEarHalfWidth(atV v: Double) -> Double {
        let m = LeafSilhouette.bunnyEarPad.margin(tier: .hero, subdivisions: 3)
        for i in 0 ..< m.count - 1 where v >= m[i].v && v <= m[i + 1].v {
            let span = m[i + 1].v - m[i].v
            return span < 1e-9 ? m[i + 1].u : m[i].u + (m[i + 1].u - m[i].u) * (v - m[i].v) / span
        }
        return 0
    }

    /// The pads — the single source the mesh and its tests read.
    public static func bunnyEarsPads(spec: GardenPlantSpec) -> [BunnyEarPad] {
        var rng = speciesRNG(spec, lane: 0xB0E5)
        let up = Vec3(0, 1, 0)
        let target = max(5, min(16, Int((9.0 * Double(spec.foliageDensity)).rounded())))
        let baseHeight = spec.size * 0.36

        func make(parent: Int?, depth: Int, joint: Vec3, grow: Vec3, normalHint: Vec3,
                  height: Double) -> BunnyEarPad {
            var y = normalize3(grow)
            if y.y < 0.25 { y = normalize3(y + up * ((0.25 - y.y) * 2)) }
            var n = normalHint - y * dot3(normalHint, y)
            if len3(n) < 1e-3 { n = frame(yAxis: y).x }
            n = normalize3(n)
            let aspect = 0.70 + rng.unit() * 0.22
            let cup = (rng.unit() - 0.5) * 0.16
            let bow = (rng.unit() - 0.5) * 0.12
            let l1 = rng.unit() * 0.07, p1 = rng.unit() * 2 * Double.pi
            let l2 = rng.unit() * 0.05, p2 = rng.unit() * 2 * Double.pi
            var lumps: [SIMD4<Double>] = []
            for _ in 0 ..< 2 {
                let x = rng.unit() * 1.2 - 0.6, yy = rng.unit() - 0.4, a = 0.3 + rng.unit() * 0.5
                lumps.append(SIMD4(x, yy, a, rng.unit() < 0.5 ? -1 : 1))
            }
            let wrinkle = depth == 0 ? 0.6 + rng.unit() * 0.4 : (depth == 1 ? rng.unit() * 0.3 : 0)
            return BunnyEarPad(parent: parent, depth: depth,
                               joint: joint - y * (height * 0.10),   // embedded: a neck, never a gap
                               yAxis: y, normal: n, height: height, width: height * aspect,
                               thickness: height * 0.07, cup: cup, bow: bow,
                               lobe1: l1, lobe1Phase: p1, lobe2: l2, lobe2Phase: p2,
                               lumps: lumps, wrinkle: wrinkle)
        }

        var pads: [BunnyEarPad] = []
        let roots = spec.foliageDensity >= 1.3 ? 2 : 1
        let start = rng.unit() * 2 * Double.pi
        for i in 0 ..< roots {
            let az = start + Double(i) * Double.pi * 0.9
            let out = Vec3(cos(az), 0, sin(az))
            let lean = 0.08 + rng.unit() * 0.22
            let facing = rng.unit() * 2 * Double.pi
            let joint = out * (roots == 1 ? 0 : spec.size * 0.07)
            pads.append(make(parent: nil, depth: 0, joint: joint, grow: up * cos(lean) + out * sin(lean),
                             normalHint: Vec3(cos(facing), 0, sin(facing)),
                             height: baseHeight * (0.9 + rng.unit() * 0.2)))
        }

        var next = 0
        while pads.count < target && next < pads.count {
            let p = pads[next], index = next
            next += 1
            guard p.depth < 3 else { continue }
            let kids = p.depth == 0 ? 2 + (rng.unit() < 0.5 ? 1 : 0) : 1 + (rng.unit() < 0.55 ? 1 : 0)
            var side: Double = rng.unit() < 0.5 ? -1 : 1
            let x = p.xAxis
            for _ in 0 ..< kids where pads.count < target {
                let fromFace = rng.unit() < 0.25
                let v = fromFace ? 0.5 + rng.unit() * 0.25 : 0.55 + rng.unit() * 0.40
                let halfW = p.width / 2 * bunnyEarHalfWidth(atV: v)
                let face: Double = rng.unit() < 0.5 ? -1 : 1
                let joint: Vec3, grow: Vec3
                if fromFace {
                    let surface = p.thickness * 0.5 * (1 + PadConstructor.defaultCrown)
                    joint = p.joint + p.yAxis * (p.height * v) + x * (side * halfW * (0.2 + rng.unit() * 0.3))
                        + p.normal * (face * surface * 0.6)
                    grow = p.normal * (face * 0.75) + up * 0.6 + x * (side * 0.2)
                } else {
                    joint = p.joint + p.yAxis * (p.height * v) + x * (side * halfW * 0.92)
                    grow = x * (side * (1.1 - v)) + p.yAxis * v + p.normal * ((rng.unit() - 0.5) * 0.7)
                }
                let twist = (rng.unit() < 0.5 ? -1.0 : 1.0) * (0.45 + rng.unit() * 0.8)
                let height = p.height * (0.62 + rng.unit() * 0.22)
                pads.append(make(parent: index, depth: p.depth + 1, joint: bunnyEarsDisplace(p, joint),
                                 grow: grow, normalHint: p.normal * cos(twist) + x * sin(twist),
                                 height: height))
                side = -side
            }
        }
        return pads
    }

    /// The monstrose deformity of pad `p` applied to a point of its flat, symmetric build: an
    /// outline wobble, a cup across and a bow along the pad, two lumps, and wrinkles on old pads.
    /// Everything is windowed to zero at the joint, so a pad stays where its parent holds it. The
    /// lumps and wrinkles push each face along its OWN outward side and never by more than the pad's
    /// thickness, so a face can never be pushed through the other one.
    public static func bunnyEarsDisplace(_ p: BunnyEarPad, _ q: Vec3) -> Vec3 {
        let x = p.xAxis
        let d = q - p.joint
        let along = dot3(d, p.yAxis), across = dot3(d, x), off = dot3(d, p.normal)
        let v = along / p.height
        let xl = across / (p.width / 2), yl = (v - 0.5) * 2
        let neck = smoothstep(0.10, 0.40, v)
        let ang = atan2(yl, xl)
        let stretch = 1 + neck * (p.lobe1 * sin(ang + p.lobe1Phase) + p.lobe2 * sin(2 * ang + p.lobe2Phase))
        let mid = p.height * 0.5
        var out = p.joint + p.yAxis * (mid + (along - mid) * stretch) + x * (across * stretch) + p.normal * off
        out = out + p.normal * (neck * p.height * (p.cup * xl * xl + p.bow * yl * yl))
        let side: Double = off >= 0 ? 1 : -1
        for l in p.lumps where l.w == side {
            let r2 = (xl - l.x) * (xl - l.x) + (yl - l.y) * (yl - l.y)
            out = out + p.normal * (side * l.z * p.thickness * exp(-r2 / 0.18) * neck)
        }
        if p.wrinkle > 0 {
            out = out + p.normal * (side * p.wrinkle * p.thickness * 0.18 * sin(yl * 7 + xl * 1.3) * neck)
        }
        return out
    }

    public static func bunnyEarsPadMesh(_ p: BunnyEarPad) -> Mesh3 {
        var m = Mesh3()
        PadConstructor.emitPad(into: &m,
                               placement: .init(position: p.joint + p.yAxis * (p.height * 0.46),
                                                xAxis: p.xAxis, yAxis: p.yAxis, bentNormal: p.normal,
                                                width: p.width, height: p.height),
                               silhouette: .bunnyEarPad, subdivisions: p.depth <= 1 ? 3 : 2,
                               thickness: p.thickness, winding: .singleSided)
        m.positions = m.positions.map { bunnyEarsDisplace(p, $0) }
        // A wide crease: the rim band blends into both faces, so the edge reads ROUNDED edge-on.
        return m.smoothed(creaseDegrees: 100)
    }

    /// The glochid tufts of one pad: a staggered lattice on both faces, on the deformed surface.
    public static func bunnyEarsTufts(_ p: BunnyEarPad) -> [(point: Vec3, normal: Vec3)] {
        let x = p.xAxis
        let spacing = p.height / 5.5
        var out: [(point: Vec3, normal: Vec3)] = []
        for face in [1.0, -1.0] {
            var row = 0
            var along = p.height * 0.14
            while along < p.height * 0.90 {
                let v = along / p.height
                let fullHalf = p.width / 2 * bunnyEarHalfWidth(atV: v)
                let shift = row % 2 == 0 ? 0 : spacing / 2
                let reach = Int(fullHalf / spacing) + 1
                for k in -reach ... reach {
                    let across = Double(k) * spacing + shift
                    guard abs(across) <= fullHalf * 0.80, fullHalf > 1e-6 else { continue }
                    let u = abs(across) / fullHalf
                    let surface = p.thickness * 0.5
                        + PadConstructor.defaultCrown * p.thickness * (1 - u * u) * sin(v * Double.pi)
                    let flat = p.joint + p.yAxis * along + x * across + p.normal * (face * surface)
                    out.append((bunnyEarsDisplace(p, flat), p.normal * face))
                }
                along += spacing * 0.866
                row += 1
            }
        }
        return out
    }

    /// A glochid tuft: a low four-sided cushion, its foot sunk into the pad.
    public static func appendTuft(_ m: inout Mesh3, at c: Vec3, normal n: Vec3, radius a: Double) {
        let f = frame(yAxis: n)
        let base = c - n * (a * 0.35)
        let tip = c + n * (a * 0.7)
        let ring = (0 ..< 4).map { i -> Vec3 in
            let t = Double(i) * Double.pi / 2
            return base + (f.x * cos(t) + f.z * sin(t)) * a
        }
        for i in 0 ..< 4 {
            let p = ring[i], q = ring[(i + 1) % 4]
            m.addTriangle(p, q, tip, outward: (p + q) * 0.5 - base)
        }
    }

    /// The flowers: one to three, straight on the tip rim of the youngest pads.
    public static func bunnyEarsFlowers(spec: GardenPlantSpec, pads: [BunnyEarPad]) -> [BunnyEarsFlower] {
        var rng = speciesRNG(spec, lane: 0xB0E7)
        let count = 1 + (rng.unit() < 0.45 ? 1 : 0) + (rng.unit() < 0.2 ? 1 : 0)
        let youngest = pads.indices.sorted { (pads[$0].depth, $0) > (pads[$1].depth, $1) }
        return youngest.prefix(count).map { i in
            let p = pads[i]
            let v = 0.90
            let across = (rng.unit() - 0.5) * p.width * bunnyEarHalfWidth(atV: v) * 0.5
            let point = bunnyEarsDisplace(p, p.joint + p.yAxis * (p.height * v) + p.xAxis * across)
            let axis = normalize3(p.yAxis + p.normal * ((rng.unit() - 0.5) * 0.6) + Vec3(0, 0.4, 0))
            return BunnyEarsFlower(pad: i, point: point, axis: axis)
        }
    }

    public static func bunnyEarsParts(spec: GardenPlantSpec) -> Parts {
        let pads = bunnyEarsPads(spec: spec)
        var rng = speciesRNG(spec, lane: 0xB0E6)
        var body = Mesh3(), tufts = Mesh3()
        var groups = SpeciesColorGroups()
        for p in pads {
            body.append(bunnyEarsPadMesh(p))
            for t in bunnyEarsTufts(p) {
                appendTuft(&tufts, at: t.point, normal: t.normal, radius: p.height * bunnyEarsTuftRadius)
            }
        }

        let petalLen = max(0.012, min(0.028, spec.size * 0.045))
        var stamens = Mesh3()
        for flower in bunnyEarsFlowers(spec: spec, pads: pads) {
            // The receptacle is pad tissue — it wears the pad's colour.
            let ro = petalLen * 0.28
            body.append(revolvedPart([(ro * 0.45, -ro * 0.4), (ro * 0.8, ro * 0.5), (ro, ro * 1.5), (ro * 0.9, ro * 1.9)],
                                     segments: 8, origin: flower.point, axis: flower.axis))
            let center = flower.point + flower.axis * (ro * 1.8)
            let f = frame(yAxis: flower.axis)
            let phase = rng.unit() * Double.pi
            for k in 0 ..< 10 {
                let a = phase + Double(k) * Double.pi / 5 + (rng.unit() - 0.5) * 0.2
                let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                emitBandedPetal(into: &groups,
                                placement: petalPlacement(center: center, axis: flower.axis, radial: radial,
                                                          pitch: 0.55 + (k % 2 == 0 ? 0 : 0.14),
                                                          length: petalLen * (0.9 + rng.unit() * 0.2),
                                                          width: petalLen * 0.5),
                                silhouette: .cactusPetal, subdivisions: 2, fold: 0.22, curl: -0.06,
                                bands: [PetalBand(vEnd: 0.3, color: bunnyEarsThroatColor),
                                        PetalBand(vEnd: 1.0, color: bunnyEarsFlowerColor)])
            }
            let rs = petalLen * 0.2
            stamens.append(revolvedPart([(rs, 0), (rs * 0.9, rs * 0.6), (0, rs * 0.9)],
                                        segments: 8, origin: center, axis: flower.axis))
        }
        groups.append(stamens, color: bunnyEarsStamenColor)
        return Parts(stem: body, foliage: tufts, blooms: groups.groups)
    }
}
