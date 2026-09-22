import Foundation
import simd
import VisualizerMaterials

extension LeafSilhouette {
    /// A yucca's sword leaf: parallel-sided for most of its length, then a quick taper to the point.
    public static let yuccaSword = LeafSilhouette(
        name: "yuccaSword",
        hero: [(0.00, 0.00), (0.06, 0.85), (0.45, 1.00), (0.80, 0.70), (1.00, 0.00)])
}

/// *Yucca gigantea* — the giant (spineless) yucca, the yucca TREE of Los Angeles front yards, whose
/// white flowers are eaten and steeped.
///
/// What makes it read as this plant, trait by trait:
///  - **A swollen "elephant foot" base** breaking into a few thick, bare, grey-brown trunks that
///    lean apart and fork once or twice — never a crown of twigs.
///  - **Every trunk ends in a HEAD**: a dense spiral of stiff sword leaves, the young ones upright,
///    the middle ones radiating, so the head is a sphere of blades rather than a canopy.
///  - **A skirt of spent leaves** hanging straight down the trunk below each head.
///  - **A white panicle** standing out of the top of some heads: a tall cone of cream bells.
///
/// `spec.size` is the height to the top of the tallest head; trunks, heads and blade length all
/// derive from it. Counts ride `foliageDensity`, arrangement rides `seed`.
extension GardenPlantMesh {

    /// One leaf head — the single source the mesh and its tests read.
    public struct YuccaHead: Sendable, Equatable {
        public var centre: Vec3
        public var axis: Vec3
        public var bladeLength: Double
        public var inFlower: Bool
    }

    /// The trunks as polylines (ground → head), and the head each one carries.
    public static func giantYuccaSkeleton(spec: GardenPlantSpec) -> (trunks: [[Vec3]], heads: [YuccaHead]) {
        var rng = speciesRNG(spec, lane: 0x70CCA)
        let h = spec.size, up = Vec3(0, 1, 0)
        let stems = max(2, min(4, Int((2.6 + rng.unit() * 1.2).rounded())))
        let start = rng.unit() * 2 * Double.pi
        var trunks: [[Vec3]] = [], heads: [YuccaHead] = []
        func head(at p: Vec3, along dir: Vec3, index: Int) {
            heads.append(YuccaHead(centre: p, axis: normalize3(dir + up * 0.6),
                                   bladeLength: min(1.1, max(0.35, h * (0.19 + rng.unit() * 0.03))),
                                   inFlower: index % 3 == 0))
        }
        for i in 0 ..< stems {
            let az = start + Double(i) * 2 * Double.pi / Double(stems) + (rng.unit() - 0.5) * 0.6
            let out = Vec3(cos(az), 0, sin(az))
            let top = h * (0.62 + rng.unit() * 0.22)                 // where this trunk's head sits
            let lean = h * (0.10 + rng.unit() * 0.12)
            let foot = out * (h * 0.035)
            let knee = foot + out * (lean * 0.35) + up * (top * 0.45)
            let fork = foot + out * (lean * 0.75) + up * (top * 0.78)
            if rng.unit() < 0.55 {
                // Fork: two heads off one trunk.
                let side = Vec3(-out.z, 0, out.x)
                for s in [-1.0, 1.0] {
                    let tip = fork + (out * 0.4 + side * s) * (h * 0.10) + up * (top * (0.20 + rng.unit() * 0.10))
                    trunks.append([foot, knee, fork, tip])
                    head(at: tip, along: tip - fork, index: heads.count)
                }
            } else {
                let tip = fork + out * (lean * 0.25) + up * (top * 0.22)
                trunks.append([foot, knee, fork, tip])
                head(at: tip, along: tip - fork, index: heads.count)
            }
        }
        return (trunks, heads)
    }

    public static let yuccaSpentLeaf = Vec3(0.52, 0.44, 0.27)
    public static let yuccaFlower = Vec3(0.93, 0.91, 0.82)

    public static func giantYuccaParts(spec: GardenPlantSpec) -> Parts {
        let h = spec.size, up = Vec3(0, 1, 0)
        var rng = speciesRNG(spec, lane: 0x70CCB)
        let skeleton = giantYuccaSkeleton(spec: spec)
        var stem = Mesh3(), foliage = Mesh3()
        var groups = SpeciesColorGroups()

        // ── The elephant foot, and the trunks out of it ──
        let trunkR = h * 0.030
        stem.append(revolvedPart([(0, 0), (trunkR * 3.4, 0), (trunkR * 3.0, h * 0.03), (trunkR * 2.0, h * 0.08),
                                  (trunkR * 1.3, h * 0.13), (0, h * 0.15)],
                                 segments: 14, origin: Vec3(0, -0.02, 0), axis: up))
        for path in skeleton.trunks {
            var t = Mesh3()
            if t.sweep(profile: .circle(radius: trunkR, segments: 8), along: path,
                       scales: [1.35, 1.0, 0.85, 0.75], capStart: false, capEnd: true) {
                stem.append(t.smoothed(creaseDegrees: 80))
            }
        }

        // ── Heads ──
        let perHead = max(24, min(50, Int((44.0 * Double(spec.foliageDensity)).rounded())))
        for head in skeleton.heads {
            let f = frame(yAxis: head.axis)
            let phase = rng.unit() * 2 * Double.pi
            var live = Mesh3(), spent = Mesh3()
            for i in 0 ..< perHead {
                let t = Double(i) / Double(perHead - 1)                // 0 innermost/upright … 1 outermost
                let az = phase + Double(i) * Phyllotaxis.goldenAngle
                let radial = normalize3(f.x * cos(az) + f.z * sin(az))
                // Pitch off the head axis: a spear at the centre, flat out at the equator, and the
                // last fifth hanging below it — those are the spent skirt.
                let pitch = 0.12 + t * 1.75 + (rng.unit() - 0.5) * 0.18
                let isSpent = t > 0.82
                let dir = isSpent ? normalize3(radial * 0.35 - up) : normalize3(head.axis * cos(pitch) + radial * sin(pitch))
                let len = head.bladeLength * (isSpent ? 0.85 : 0.75 + rng.unit() * 0.35)
                let side = normalize3(cross3(dir, radial + head.axis * 0.01))
                let face = normalize3(cross3(side, dir))
                let base = head.centre + radial * (trunkR * 0.7) - head.axis * (t * head.bladeLength * 0.22)
                var blade = Mesh3()
                LeafConstructor.emitBlade(into: &blade,
                                          placement: .init(position: base + dir * (len * 0.46), xAxis: side, yAxis: dir,
                                                           bentNormal: face, width: len * 0.10, height: len),
                                          silhouette: .yuccaSword, subdivisions: 1,
                                          fold: 0.22, curl: isSpent ? 0.05 : 0.06 + t * 0.22, winding: .doubleSidedShell)
                if isSpent { spent.append(blade) } else { live.append(blade) }
            }
            foliage.append(live.smoothed())
            groups.append(spent.smoothed(), color: yuccaSpentLeaf)

            // ── The panicle: a stalk out of the head's heart, hung with cream bells ──
            if head.inFlower {
                let stalkTop = head.centre + head.axis * (head.bladeLength * 1.25)
                var stalk = Mesh3()
                if stalk.sweep(profile: .circle(radius: trunkR * 0.22, segments: 5),
                               along: [head.centre, head.centre + head.axis * (head.bladeLength * 0.6), stalkTop],
                               scales: [1, 0.8, 0.4]) {
                    groups.append(stalk.smoothed(), color: spec.leafColor)
                }
                var bells = Mesh3()
                let tiers = 7
                for k in 0 ..< tiers {
                    let u = Double(k) / Double(tiers - 1)
                    let at = head.centre + head.axis * (head.bladeLength * (0.55 + 0.68 * u))
                    let ring = head.bladeLength * 0.20 * (1 - u * 0.85)
                    let count = max(2, Int((6.0 * (1 - u * 0.7)).rounded()))
                    for j in 0 ..< count {
                        let a = Double(j) / Double(count) * 2 * Double.pi + u * 2.4
                        let p = at + (f.x * cos(a) + f.z * sin(a)) * ring - up * 0.02
                        let r = head.bladeLength * 0.035
                        bells.append(revolvedPart([(0, 0), (r, r * 0.5), (r * 0.8, r * 1.6), (0, r * 2.1)],
                                                  segments: 5, origin: p, axis: up * -1))
                    }
                }
                groups.append(bells, color: yuccaFlower)
            }
        }
        return Parts(stem: stem, foliage: foliage, blooms: groups.groups)
    }
}
