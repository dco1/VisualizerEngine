import Foundation
import simd
import VisualizerMaterials

/// *Bougainvillea glabra* (DH-0796). A THORNY ARCHING VINE whose colour is papery bracts.
///
/// What makes it read as this plant, trait by trait:
///  - **Arching woody canes, not an upright clump.** Each cane rises steeply from the crown, tops
///    out, and arches over to hang its tip back down (`bougainvilleaCanePoint`). Grown without a
///    wall or trellis, that is the fountain a real bougainvillea makes.
///  - **Grey-brown old wood with sharp thorns** (`bougainvilleaThorns`) along the woody run, and
///    **reddish new growth** at every cane tip (a colour group past `bougainvilleaWoodyFraction`).
///  - **The colour is BRACTS in TRIOS** (`bougainvilleaTrios`): three broad, thin, crinkled papery
///    sheets round tiny cream true flowers, gathered into panicles along the outer cane only — so
///    the colour masses out toward the tips while the cane near the crown stays bare
///    (`bougainvilleaBloomFrom`).
///  - **Two-toned on one plant**: most trios magenta, some pale blush at the base shading to magenta.
///  - **Sparse simple ovate leaves** — the bracts, not the leaves, carry the volume.
///  - Bracts ride the blooms part, so the bridge's outdoor thin-sheet backlight makes them glow.
///
/// Draping OVER a wall, fence or arbor needs the plant to know its neighbours, which a garden plant's
/// mesh (built from its spec alone) does not; this is the free-standing habit.
extension GardenPlantMesh {

    public struct BougainvilleaCane: Sendable, Equatable {
        public var azimuth: Double
        /// Plan offset of the cane's foot from the crown centre (metres).
        public var footRadius: Double
        /// Height of the top of the arch (metres).
        public var peak: Double
        /// Fraction of the cane's run at which it tops out.
        public var peakAt: Double
        /// Height the tip hangs back down to (metres).
        public var tipHeight: Double
        /// Plan distance of the tip from the crown (metres).
        public var reach: Double
        /// Sideways wander, as a fraction of `reach`.
        public var sway: Double
    }

    public struct BougainvilleaThorn: Sendable, Equatable {
        public var t: Double
        public var base: Vec3
        public var tip: Vec3
    }

    public struct BougainvilleaTrio: Sendable, Equatable {
        public var cane: Int
        public var t: Double
        public var center: Vec3
        public var axis: Vec3
        public var phase: Double
        public var variegated: Bool
    }

    public static let bougainvilleaStations = 13
    /// Along a cane's run, old wood hands over to red new growth here.
    public static let bougainvilleaWoodyFraction = 0.75
    /// No bract trio sits closer to the crown than this fraction of a cane's run.
    public static let bougainvilleaBloomFrom = 0.45
    public static let bougainvilleaBractsPerTrio = 3
    /// Trios gather into panicles of this many.
    public static let bougainvilleaTriosPerPanicle = 3

    public static let bougainvilleaMagenta   = Vec3(0.80, 0.10, 0.46)
    public static let bougainvilleaBlush     = Vec3(0.95, 0.80, 0.88)
    public static let bougainvilleaFloret    = Vec3(0.96, 0.94, 0.82)
    public static let bougainvilleaNewGrowth = Vec3(0.46, 0.16, 0.12)

    /// The canes — the single source the mesh, the thorns, the trios and the tests read.
    public static func bougainvilleaCanes(spec: GardenPlantSpec) -> [BougainvilleaCane] {
        var rng = speciesRNG(spec, lane: 0xB060)
        let count = max(5, min(10, Int((7.0 * Double(spec.foliageDensity)).rounded())))
        let start = rng.unit() * 2 * Double.pi
        return (0 ..< count).map { i in
            let peak = spec.size * (0.72 + rng.unit() * 0.22)
            return BougainvilleaCane(
                azimuth: start + Double(i) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.4,
                footRadius: spec.size * (0.01 + rng.unit() * 0.03),
                peak: peak,
                peakAt: 0.40 + rng.unit() * 0.16,
                tipHeight: peak * (0.30 + rng.unit() * 0.30),
                reach: spec.planHalfWidth * (0.62 + rng.unit() * 0.30),
                sway: (rng.unit() - 0.5) * 0.24)
        }
    }

    /// A point on a cane at `t` (0 = the foot, 1 = the hanging tip): steeply up to the arch, then
    /// over and down, the plan distance growing all the way out.
    public static func bougainvilleaCanePoint(_ c: BougainvilleaCane, t: Double) -> Vec3 {
        let out = Vec3(cos(c.azimuth), 0, sin(c.azimuth)), side = Vec3(-sin(c.azimuth), 0, cos(c.azimuth))
        let y = t <= c.peakAt
            ? c.peak * sin(t / c.peakAt * Double.pi / 2)
            : c.tipHeight + (c.peak - c.tipHeight) * cos((t - c.peakAt) / (1 - c.peakAt) * Double.pi / 2)
        let d = c.footRadius + (c.reach - c.footRadius) * pow(t, 1.15)
        return out * d + side * (c.reach * c.sway * sin(t * Double.pi)) + Vec3(0, y, 0)
    }

    public static func bougainvilleaCaneTangent(_ c: BougainvilleaCane, t: Double) -> Vec3 {
        let a = bougainvilleaCanePoint(c, t: max(0, t - 0.01)), b = bougainvilleaCanePoint(c, t: min(1, t + 0.01))
        return normalize3(b - a)
    }

    public static func bougainvilleaCaneRadius(size: Double, t: Double) -> Double {
        size * (0.0065 + (0.0018 - 0.0065) * t)
    }

    /// The thorns along one cane's woody run — each a small cone standing off the bark, raked toward
    /// the tip, spiralling round the cane.
    public static func bougainvilleaThorns(_ c: BougainvilleaCane, size: Double) -> [BougainvilleaThorn] {
        var thorns: [BougainvilleaThorn] = []
        var t = 0.10, k = 0
        while t < bougainvilleaWoodyFraction - 0.02 {
            let p = bougainvilleaCanePoint(c, t: t), tangent = bougainvilleaCaneTangent(c, t: t)
            let f = frame(yAxis: tangent)
            let a = Double(k) * Phyllotaxis.goldenAngle + c.azimuth
            let radial = normalize3(f.x * cos(a) + f.z * sin(a))
            let base = p + radial * (bougainvilleaCaneRadius(size: size, t: t) * 0.8)
            let dir = normalize3(radial * 0.85 + tangent * 0.45)
            thorns.append(BougainvilleaThorn(t: t, base: base, tip: base + dir * (size * 0.011)))
            t += 0.075
            k += 1
        }
        return thorns
    }

    /// The bract trios, gathered into panicles along the outer run of every cane, crowding the tips.
    public static func bougainvilleaTrios(spec: GardenPlantSpec) -> [BougainvilleaTrio] {
        let canes = bougainvilleaCanes(spec: spec)
        var rng = speciesRNG(spec, lane: 0xB062)
        let up = Vec3(0, 1, 0)
        let total = max(18, min(66, Int((40.0 * Double(spec.foliageDensity)).rounded())))
        var trios: [BougainvilleaTrio] = []
        for (ci, cane) in canes.enumerated() {
            let n = total / canes.count + (ci < total % canes.count ? 1 : 0)
            let panicles = max(1, (n + bougainvilleaTriosPerPanicle - 1) / bougainvilleaTriosPerPanicle)
            for j in 0 ..< n {
                let panicle = j / bougainvilleaTriosPerPanicle
                let u = (Double(panicle) + 0.5 + (rng.unit() - 0.5) * 0.4) / Double(panicles)
                let t = min(0.99, bougainvilleaBloomFrom + (1 - bougainvilleaBloomFrom) * pow(max(0, u), 0.6)
                                  + (rng.unit() - 0.5) * 0.03)
                let p = bougainvilleaCanePoint(cane, t: t), tangent = bougainvilleaCaneTangent(cane, t: t)
                let f = frame(yAxis: tangent)
                let a = rng.unit() * 2 * Double.pi
                let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                let dir = normalize3(radial + up * 0.7)
                trios.append(BougainvilleaTrio(cane: ci, t: t,
                                               center: p + dir * (spec.size * (0.012 + rng.unit() * 0.02)),
                                               axis: normalize3(dir + up * 0.3),
                                               phase: rng.unit() * 2 * Double.pi,
                                               variegated: rng.unit() < 0.3))
            }
        }
        // Two-toned on one plant, always: the first trio plain, the second blushed.
        if trios.count >= 2 { trios[0].variegated = false; trios[1].variegated = true }
        return trios
    }

    public static func bougainvilleaParts(spec: GardenPlantSpec) -> Parts {
        let size = spec.size
        let up = Vec3(0, 1, 0)
        var wood = Mesh3(), newGrowth = Mesh3(), foliage = Mesh3()
        var groups = SpeciesColorGroups()
        var rng = speciesRNG(spec, lane: 0xB061)
        let canes = bougainvilleaCanes(spec: spec)
        let stations = (0 ..< bougainvilleaStations).map { Double($0) / Double(bougainvilleaStations - 1) }
        let woodyEnd = Int((Double(bougainvilleaStations - 1) * bougainvilleaWoodyFraction).rounded())
        let rBase = bougainvilleaCaneRadius(size: size, t: 0)

        for cane in canes {
            let path = stations.map { bougainvilleaCanePoint(cane, t: $0) }
            let scales = stations.map { bougainvilleaCaneRadius(size: size, t: $0) / rBase }
            var old = Mesh3()
            if old.sweep(profile: .circle(radius: rBase, segments: 5), along: Array(path[...woodyEnd]),
                         scales: Array(scales[...woodyEnd]), capStart: false, capEnd: false) {
                wood.append(old.smoothed())
            }
            var young = Mesh3()
            if young.sweep(profile: .circle(radius: rBase, segments: 5), along: Array(path[woodyEnd...]),
                           scales: Array(scales[woodyEnd...]), capStart: false) {
                newGrowth.append(young.smoothed())
            }

            for thorn in bougainvilleaThorns(cane, size: size) {
                emitThorn(&wood, base: thorn.base, tip: thorn.tip,
                          radius: bougainvilleaCaneRadius(size: size, t: thorn.t) * 0.5)
            }

            // Sparse leaves, alternate along the cane.
            let leaves = max(3, min(5, Int((4.0 * Double(spec.foliageDensity)).rounded())))
            for k in 0 ..< leaves {
                let t = 0.28 + 0.66 * (Double(k) + 0.5) / Double(leaves) + (rng.unit() - 0.5) * 0.04
                let p = bougainvilleaCanePoint(cane, t: t), tangent = bougainvilleaCaneTangent(cane, t: t)
                var across = cross3(tangent, up)
                across = len3(across) > 1e-6 ? normalize3(across) : Vec3(1, 0, 0)
                let grow = normalize3(tangent * 0.55 + across * (k % 2 == 0 ? 0.8 : -0.8) + up * 0.25)
                var face = up - grow * dot3(up, grow)
                face = len3(face) > 1e-6 ? normalize3(face) : across
                let len = size * (0.05 + rng.unit() * 0.02)
                var leaf = Mesh3()
                BladeDescription.bougainvilleaLeaf.emit(into: &leaf, position: p + grow * (len * 0.46),
                                                        xAxis: normalize3(cross3(grow, face)), yAxis: grow,
                                                        bentNormal: face, length: len, winding: .doubleSidedShell)
                foliage.append(leaf.smoothed())
            }
        }

        // Bract trios round their tiny cream flowers.
        var florets = Mesh3()
        for trio in bougainvilleaTrios(spec: spec) {
            let f = frame(yAxis: trio.axis)
            var len = 0.0
            for k in 0 ..< bougainvilleaBractsPerTrio {
                let a = trio.phase + Double(k) * 2 * Double.pi / Double(bougainvilleaBractsPerTrio)
                let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                len = size * (0.038 + rng.unit() * 0.008)
                let bands = trio.variegated
                    ? [PetalBand(vEnd: 0.45, color: bougainvilleaBlush), PetalBand(vEnd: 1.0, color: bougainvilleaMagenta)]
                    : [PetalBand(vEnd: 1.0, color: bougainvilleaMagenta)]
                // Paper: a fine crinkle across the sheet.
                let crinkle = len * 0.05, crinklePhase = rng.unit() * 2 * Double.pi
                let center = trio.center, axis = trio.axis, fx = f.x, fz = f.z, l = len
                emitBandedPetal(into: &groups,
                                placement: petalPlacement(center: center, axis: axis, radial: radial,
                                                          pitch: 0.60 + (rng.unit() - 0.5) * 0.2,
                                                          length: len, width: len * 0.95),
                                silhouette: .bougainvilleaBract, subdivisions: 1, fold: 0.34, curl: -0.08,
                                bands: bands, waveAmplitude: 0.06, wavePhase: rng.unit() * 6,
                                displace: { p in
                                    let d = p - center
                                    let planar = d - axis * dot3(d, axis)
                                    let r = len3(planar)
                                    guard r > 1e-6 else { return p }
                                    let ang = atan2(dot3(planar, fz), dot3(planar, fx))
                                    return p + axis * (crinkle * (r / l) * sin(ang * 9 + (r / l) * 13 + crinklePhase))
                                })
            }
            // The true flowers: a tiny cream speck in the throat of the trio.
            PottedPlantMesh.emitSprig(petals: &florets, at: trio.center, faceDir: trio.axis,
                                      scale: len * 0.35, rng: &rng)
        }
        groups.append(florets, color: bougainvilleaFloret)

        var blooms = SpeciesColorGroups()
        blooms.append(newGrowth, color: bougainvilleaNewGrowth)
        for g in groups.groups { blooms.append(g.mesh, color: g.color) }
        return Parts(stem: wood, foliage: foliage, blooms: blooms.groups)
    }

    /// A small three-sided cone from `base` to `tip` — a thorn, a spur. Open at the base, which
    /// sits against the bark.
    public static func emitThorn(_ mesh: inout Mesh3, base: Vec3, tip: Vec3, radius: Double) {
        let axis = normalize3(tip - base)
        let f = frame(yAxis: axis)
        let ring = (0 ..< 3).map { k -> Vec3 in
            let a = Double(k) * 2 * Double.pi / 3
            return base + (f.x * cos(a) + f.z * sin(a)) * radius
        }
        let core = base + (tip - base) / 3
        for k in 0 ..< 3 {
            let a = ring[k], b = ring[(k + 1) % 3]
            mesh.addTriangle(a, b, tip, outward: normalize3((a + b + tip) / 3 - core))
        }
    }
}

extension LeafSilhouette {
    /// A bougainvillea bract: a broad ovate papery sheet, widest just below the middle. Three
    /// controls — a plant carries well over a hundred.
    public static let bougainvilleaBract = LeafSilhouette(
        name: "bougainvilleaBract",
        hero: [(0.00, 0.00), (0.42, 1.00), (1.00, 0.00)])

    /// A bougainvillea leaf: simple ovate, widest below the middle, a pointed tip.
    public static let bougainvilleaLeaf = LeafSilhouette(
        name: "bougainvilleaLeaf",
        hero: [(0.00, 0.00), (0.36, 1.00), (0.74, 0.66), (1.00, 0.00)])
}

extension BladeDescription {
    /// Thin, soft, a gentle droop.
    public static let bougainvilleaLeaf = BladeDescription(
        silhouette: .bougainvilleaLeaf, aspect: 0.58, fold: 0.22, curl: 0.12, subdivisions: 1)
}
