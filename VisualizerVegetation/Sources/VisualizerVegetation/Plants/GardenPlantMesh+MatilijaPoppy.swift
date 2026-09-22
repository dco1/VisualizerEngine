import Foundation
import simd
import VisualizerMaterials

/// *Romneya coulteri* — the Matilija poppy, "the fried egg flower" (DH-0648). Shrub scale: a TALL
/// MULTI-STEMMED CLUMP.
///
/// What makes it read as this plant, trait by trait:
///  - **Several tall stems from one crown**, woody and grey at the base (the opaque `stem` part),
///    green above (a colour group).
///  - **Deeply lobed, pointed, grey-green leaves** on the LOWER stems only — every leaf sits inside
///    `matilijaLeafBand`, and every flower is above `matilijaFlowerMinHeight`, so the display height
///    and the leaf mass are always visibly separated by bare stem.
///  - **Few, huge flowers** (`matilijaStems`: two to four open per plant). Each is SIX broad white
///    crepe-paper petals — ruffled at the margin, radially crinkled, near flat — layered in two
///    pitches, some browned along a creased edge; round a dense golden domed stamen boss with a
///    fluffy fringe, big enough to be the "yolk".
///  - **Round silver-green buds that NOD** on the stems not yet in flower.
extension GardenPlantMesh {

    public enum MatilijaTop: Sendable, Equatable { case flower, noddingBud }

    public struct MatilijaStem: Sendable, Equatable {
        public var azimuth: Double
        public var baseRadius: Double
        /// Stem length along its lean (metres).
        public var length: Double
        /// Radians off vertical, outward.
        public var lean: Double
        public var top: MatilijaTop
    }

    /// Leaves grow only between these fractions of `size` (their attachment heights).
    public static let matilijaLeafBand = (lo: 0.10, hi: 0.46)
    /// No flower sits lower than this fraction of `size`.
    public static let matilijaFlowerMinHeight = 0.72

    public static let matilijaPetalWhite  = Vec3(0.92, 0.92, 0.89)
    public static let matilijaPetalBruise = Vec3(0.60, 0.50, 0.37)
    public static let matilijaYolk        = Vec3(0.95, 0.68, 0.08)
    public static let matilijaYolkFringe  = Vec3(0.98, 0.80, 0.24)
    public static let matilijaUpperStem   = Vec3(0.38, 0.47, 0.34)
    public static let matilijaBud         = Vec3(0.60, 0.65, 0.57)

    /// The stems — the single source both the mesh and its tests read.
    public static func matilijaStems(spec: GardenPlantSpec) -> [MatilijaStem] {
        var rng = speciesRNG(spec, lane: 0x0E66)
        let count = max(4, min(9, Int((6.0 * Double(spec.foliageDensity)).rounded())))
        let flowers = max(2, min(4, Int((Double(count) * 0.5).rounded())))
        let start = rng.unit() * 2 * Double.pi
        return (0 ..< count).map { i in
            MatilijaStem(azimuth: start + Double(i) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.4,
                         baseRadius: spec.size * (0.02 + rng.unit() * 0.05),
                         length: spec.size * (0.76 + rng.unit() * 0.24),
                         lean: 0.05 + rng.unit() * 0.13,
                         top: i < flowers ? .flower : .noddingBud)
        }
    }

    public static func matilijaPoppyParts(spec: GardenPlantSpec) -> Parts {
        let size = spec.size
        let up = Vec3(0, 1, 0)
        var woody = Mesh3()
        var greenStems = Mesh3()
        var foliage = Mesh3()
        var groups = SpeciesColorGroups()
        var rng = speciesRNG(spec, lane: 0x0E67)
        let petalLen = size * 0.075

        for s in matilijaStems(spec: spec) {
            let out = Vec3(cos(s.azimuth), 0, sin(s.azimuth))
            let dir = normalize3(up * cos(s.lean) + out * sin(s.lean))
            let base = out * s.baseRadius
            let perHeight = 1 / cos(s.lean)            // stem length per metre of height
            let woodTop = base + dir * (size * 0.30 * perHeight)

            PottedPlantMesh.appendBentStemTube(&woody, from: base, to: woodTop,
                                               rBase: size * 0.009, rTip: size * 0.007,
                                               bendDir: out, bendAmount: 0.03, sides: 6)
            let arch = s.top == .flower ? 0.04 : 0.20
            let (tip, tangent) = PottedPlantMesh.appendBentStemTube(
                &greenStems, from: woodTop - dir * 0.002, to: base + dir * s.length,
                rBase: size * 0.0068, rTip: size * 0.0045, bendDir: out, bendAmount: arch, sides: 6)

            // Leaves on the lower stem, shrinking toward the top of the band.
            let leaves = max(3, min(6, Int((4.0 * Double(spec.foliageDensity)).rounded())))
            for k in 0 ..< leaves {
                let t = (Double(k) + 0.5) / Double(leaves)
                let y = size * (matilijaLeafBand.lo + (matilijaLeafBand.hi - matilijaLeafBand.lo) * t)
                let attach = base + dir * (y * perHeight)
                let az = s.azimuth + Double(k) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.3
                let tilt = 0.95 + rng.unit() * 0.35
                let grow = normalize3(Vec3(cos(az) * sin(tilt), cos(tilt), sin(az) * sin(tilt)))
                let face = normalize3(up - grow * dot3(up, grow))
                let len = size * (0.15 - 0.05 * t) * (0.85 + rng.unit() * 0.3)
                var leaf = Mesh3()
                BladeDescription.matilijaLeaf.emit(into: &leaf, position: attach + grow * (len * 0.46),
                                                   xAxis: normalize3(cross3(grow, face)), yAxis: grow,
                                                   bentNormal: face, length: len, winding: .doubleSidedShell)
                foliage.append(leaf.smoothed())
            }

            switch s.top {
            case .flower:
                let axis = normalize3(tangent * 0.4 + up * 0.7 + out * 0.25)
                let center = tip + axis * (size * 0.004)
                let f = frame(yAxis: axis)
                let phase = rng.unit() * Double.pi
                for k in 0 ..< 6 {
                    let a = phase + Double(k) * Double.pi / 3 + (rng.unit() - 0.5) * 0.12
                    let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                    let len = petalLen * (0.9 + rng.unit() * 0.2)
                    let pitch = 1.22 + (k % 2 == 0 ? 0.05 : -0.07) + (rng.unit() - 0.5) * 0.10
                    let bands = rng.unit() < 0.35
                        ? [PetalBand(vEnd: 0.84, color: matilijaPetalWhite), PetalBand(vEnd: 1.0, color: matilijaPetalBruise)]
                        : [PetalBand(vEnd: 1.0, color: matilijaPetalWhite)]
                    // Crepe paper: radial crinkles, stronger toward the rim.
                    let crinkle = len * 0.035, crinklePhase = rng.unit() * 2 * Double.pi
                    let fx = f.x, fz = f.z
                    emitBandedPetal(into: &groups,
                                    placement: petalPlacement(center: center, axis: axis, radial: radial,
                                                              pitch: pitch, length: len, width: len * 1.05),
                                    silhouette: .matilijaPetal, subdivisions: 2, fold: 0.16, curl: -0.05,
                                    bands: bands, waveAmplitude: 0.08, wavePhase: rng.unit() * 6,
                                    displace: { p in
                                        let d = p - center
                                        let planar = d - axis * dot3(d, axis)
                                        let r = len3(planar)
                                        guard r > 1e-6 else { return p }
                                        let ang = atan2(dot3(planar, fz), dot3(planar, fx))
                                        return p + axis * (crinkle * (r / len)
                                                           * sin(ang * 22 + (r / len) * 9 + crinklePhase))
                                    })
                }
                // The yolk: a domed golden stamen boss with a fluffy fringe.
                let r = petalLen * 0.30
                groups.append(revolvedPart([(r * 0.95, 0), (r, r * 0.18), (r * 0.85, r * 0.45),
                                            (r * 0.5, r * 0.66), (0, r * 0.72)],
                                           segments: 12, origin: center + axis * (petalLen * 0.02), axis: axis),
                              color: matilijaYolk)
                var fringe = Mesh3()
                for k in 0 ..< 12 {
                    let a = Double(k) * Double.pi / 6 + rng.unit() * 0.3
                    let radial = f.x * cos(a) + f.z * sin(a)
                    PottedPlantMesh.emitSprig(petals: &fringe, at: center + radial * (r * 0.9) + axis * (r * 0.25),
                                              faceDir: normalize3(axis * 0.5 + radial * 0.9),
                                              scale: petalLen * 0.10, rng: &rng)
                }
                groups.append(fringe, color: matilijaYolkFringe)

            case .noddingBud:
                // A short neck curling over, and a round bud hanging from it.
                let neck = petalLen * 0.35
                let n1 = tip + normalize3(up * 0.6 + out * 0.8) * neck
                let n2 = n1 + normalize3(out - up * 0.4) * neck
                var neckMesh = Mesh3()
                if neckMesh.sweep(profile: .circle(radius: size * 0.0045, segments: 6), along: [tip, n1, n2],
                                  scales: [1, 0.9, 0.8]) {
                    greenStems.append(neckMesh.smoothed())
                }
                let rb = petalLen * 0.28
                groups.append(revolvedPart([(0, 0), (rb * 0.7, rb * 0.25), (rb, rb * 0.9),
                                            (rb * 0.7, rb * 1.6), (0, rb * 1.85)],
                                           segments: 10, origin: n2, axis: normalize3(out - up * 1.2)),
                              color: matilijaBud)
            }
        }

        var blooms = SpeciesColorGroups()
        blooms.append(greenStems, color: matilijaUpperStem)
        for g in groups.groups { blooms.append(g.mesh, color: g.color) }
        return Parts(stem: woody, foliage: foliage, blooms: blooms.groups)
    }
}
