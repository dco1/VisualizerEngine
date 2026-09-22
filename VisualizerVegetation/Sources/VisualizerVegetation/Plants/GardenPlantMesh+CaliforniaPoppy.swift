import Foundation
import simd
import VisualizerMaterials

/// *Eschscholzia californica* — the California poppy (DH-0648). Bedding scale: a LOW SPRAWLING MAT.
///
/// What makes it read as this plant and not the generic `.flowers` clump, trait by trait:
///  - **Foliage is thread-fine and basal.** Every leaf is a thin rachis carrying narrow rolled
///    `poppyThread` segments — feathery, never a broad blade — in a low mound at the crown.
///  - **Stems are bare and waxy.** Each flower stands on its own smooth blue-green stem that leaves
///    the crown low and turns upward (decumbent, then ascending), with no leaf below the head.
///  - **Many heads, four stages at once.** `californiaPoppyHeads` mixes OPEN cups, CLOSED furled
///    cones, upright BUDS and long thin seed-pod HORNS at different heights — never a uniform stand.
///  - **The flower is a shallow cup of FOUR petals**, two pairs pitched apart so they overlap, each a
///    broad fan cupped by fold + a negative curl, and banded deep orange at the claw → paler
///    orange-yellow at the rim. No contrasting centre disc: a small cluster of stamens.
///  - **A raised collar (the torus)** where petals meet the stem, on every open/closed/pod head.
///  - Petals are thin-sheet translucent through the bridge's outdoor SSS flag, like every bloom.
///
/// All dimensions derive from `spec.size`; counts from `spec.foliageDensity`; arrangement from
/// `spec.seed`.
extension GardenPlantMesh {

    public enum PoppyStage: Sendable, Equatable, CaseIterable { case open, closed, bud, pod }

    public struct PoppyHead: Sendable, Equatable {
        public var stage: PoppyStage
        public var azimuth: Double
        /// Plan distance of the head from the crown (metres).
        public var reach: Double
        /// Target head height above the ground (metres).
        public var height: Double
    }

    /// Stage order. Any count ≥ 5 (the minimum) carries all four stages.
    public static let poppyStagePattern: [PoppyStage] =
        [.open, .closed, .bud, .pod, .open, .open, .closed, .bud, .open, .pod, .closed, .open]

    public static let poppyPetalClaw = Vec3(0.90, 0.28, 0.03)   // deep orange at the base
    public static let poppyPetalMid  = Vec3(0.96, 0.46, 0.05)
    public static let poppyPetalRim  = Vec3(0.98, 0.68, 0.16)   // paler orange-yellow toward the tip
    public static let poppyStamen    = Vec3(0.95, 0.60, 0.12)
    public static let poppyPod       = Vec3(0.37, 0.43, 0.28)

    /// The flower heads — the single source both the mesh and its tests read.
    public static func californiaPoppyHeads(spec: GardenPlantSpec) -> [PoppyHead] {
        var rng = speciesRNG(spec, lane: 0xCA11)
        let count = max(5, min(12, Int((9.0 * Double(spec.foliageDensity)).rounded())))
        let start = rng.unit() * 2 * Double.pi
        return (0 ..< count).map { i in
            let stage = poppyStagePattern[i % poppyStagePattern.count]
            let band: (lo: Double, hi: Double)
            switch stage {
            case .open:   band = (0.62, 1.00)
            case .closed: band = (0.50, 0.85)
            case .bud:    band = (0.35, 0.65)
            case .pod:    band = (0.75, 1.00)
            }
            return PoppyHead(stage: stage,
                             azimuth: start + Double(i) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.5,
                             reach: spec.size * (0.20 + rng.unit() * 0.55),
                             height: spec.size * (band.lo + rng.unit() * (band.hi - band.lo)))
        }
    }

    public static func californiaPoppyParts(spec: GardenPlantSpec) -> Parts {
        let size = spec.size
        let up = Vec3(0, 1, 0)
        var stem = Mesh3()
        var foliage = Mesh3()
        var groups = SpeciesColorGroups()
        var rng = speciesRNG(spec, lane: 0xCA12)
        let petalLen = size * 0.085

        // ── Basal mound of finely dissected leaves ──
        let leafCount = max(8, min(18, Int((13.0 * Double(spec.foliageDensity)).rounded())))
        let leafStart = rng.unit() * 2 * Double.pi
        for i in 0 ..< leafCount {
            let az = leafStart + Double(i) * Phyllotaxis.goldenAngle + (rng.unit() - 0.5) * 0.5
            let out = Vec3(cos(az), 0, sin(az)), side = Vec3(-sin(az), 0, cos(az))
            let length = size * (0.38 + rng.unit() * 0.30)
            let rise = 0.30 + rng.unit() * 0.35
            let base = out * (size * 0.03) + up * 0.004
            let p1 = base + out * (length * 0.45) + up * (length * rise)
            let p2 = base + out * (length * 0.92) + up * (length * rise * 0.75)

            var rachis = Mesh3()
            if rachis.sweep(profile: .circle(radius: size * 0.004, segments: 4), along: [base, p1, p2],
                            scales: [1, 0.65, 0.35], capStart: false, capEnd: false) {
                foliage.append(rachis.smoothed())
            }
            func along(_ t: Double) -> (Vec3, Vec3) {
                t < 0.5 ? (base + (p1 - base) * (t / 0.5), normalize3(p1 - base))
                        : (p1 + (p2 - p1) * ((t - 0.5) / 0.5), normalize3(p2 - p1))
            }
            var threads = Mesh3()
            for (t, frac) in [(0.38, 0.30), (0.60, 0.25), (0.80, 0.19)] {
                let (at, tangent) = along(t)
                for s in [-1.0, 1.0] {
                    let dir = normalize3(tangent * 0.45 + side * (s * 0.85) + up * 0.20)
                    emitPoppyThread(&threads, from: at, direction: dir, length: length * frac)
                }
            }
            let tipTangent = normalize3(p2 - p1)
            for s in [-0.65, 0.0, 0.65] {
                emitPoppyThread(&threads, from: p2, direction: normalize3(tipTangent + side * s),
                                length: length * 0.15)
            }
            foliage.append(threads.smoothed())
        }

        // ── Heads on bare stems ──
        let stemTipR = size * 0.0045
        for head in californiaPoppyHeads(spec: spec) {
            let out = Vec3(cos(head.azimuth), 0, sin(head.azimuth))
            let crown = out * (size * 0.02 * rng.unit()) + up * 0.002
            let target = out * head.reach + up * head.height
            // Bowed UP, strongest near the head: the stem leaves the crown low and turns upward.
            let (tip, tangent) = PottedPlantMesh.appendBentStemTube(
                &stem, from: crown, to: target, rBase: size * 0.0065, rTip: stemTipR,
                bendDir: up, bendAmount: 0.12 + rng.unit() * 0.12, sides: 5)

            let collarH = size * 0.012
            func collar(_ axis: Vec3) {
                stem.append(revolvedPart([(stemTipR * 0.9, 0), (stemTipR * 2.4, collarH * 0.45),
                                          (stemTipR * 2.3, collarH * 0.8), (stemTipR * 1.2, collarH)],
                                         segments: 8, origin: tip - axis * (collarH * 0.3), axis: axis))
            }

            switch head.stage {
            case .open:
                let axis = normalize3(tangent + up * 0.6)
                collar(axis)
                let center = tip + axis * (collarH * 0.65)
                let f = frame(yAxis: axis)
                let phase = rng.unit() * Double.pi
                for k in 0 ..< 4 {
                    let a = phase + Double(k) * Double.pi / 2
                    let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                    let len = petalLen * (0.9 + rng.unit() * 0.2)
                    // Two pairs pitched apart so neighbouring petals overlap instead of z-fighting.
                    let pitch = 0.92 + (k % 2 == 0 ? 0.08 : -0.08) + (rng.unit() - 0.5) * 0.08
                    emitBandedPetal(into: &groups,
                                    placement: petalPlacement(center: center, axis: axis, radial: radial,
                                                              pitch: pitch, length: len, width: len * 1.0),
                                    silhouette: .poppyPetal, subdivisions: 2, fold: 0.30, curl: -0.12,
                                    bands: [PetalBand(vEnd: 0.35, color: poppyPetalClaw),
                                            PetalBand(vEnd: 0.70, color: poppyPetalMid),
                                            PetalBand(vEnd: 1.00, color: poppyPetalRim)])
                }
                var stamens = Mesh3()
                for _ in 0 ..< 7 {
                    let a = rng.unit() * 2 * Double.pi
                    let radial = f.x * cos(a) + f.z * sin(a)
                    PottedPlantMesh.emitSprig(petals: &stamens, at: center + radial * (petalLen * 0.05),
                                              faceDir: normalize3(axis + radial * 0.4),
                                              scale: petalLen * 0.22, rng: &rng)
                }
                groups.append(stamens, color: poppyStamen)

            case .closed:
                // Furled shut into a tight cone: petals nearly upright, rolled hard, spiralled.
                let axis = normalize3(tangent + up * 0.3)
                collar(axis)
                let center = tip + axis * (collarH * 0.65)
                let f = frame(yAxis: axis)
                let phase = rng.unit() * Double.pi
                for k in 0 ..< 4 {
                    let a = phase + Double(k) * Double.pi / 2
                    let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                    let len = petalLen * 1.05
                    emitBandedPetal(into: &groups,
                                    placement: petalPlacement(center: center + radial * (petalLen * 0.03),
                                                              axis: axis, radial: radial,
                                                              pitch: 0.10 + Double(k) * 0.02,
                                                              length: len, width: len * 0.62, twist: 0.55),
                                    silhouette: .poppyPetal, subdivisions: 1, fold: 0.9, curl: 0.02,
                                    bands: [PetalBand(vEnd: 0.5, color: poppyPetalClaw),
                                            PetalBand(vEnd: 1.0, color: poppyPetalMid)])
                }

            case .bud:
                // An upright pointed cap, the same waxy green as the stem.
                let rb = petalLen * 0.20, hb = petalLen * 0.75
                stem.append(revolvedPart([(0, 0), (rb * 0.8, hb * 0.15), (rb, hb * 0.40),
                                          (rb * 0.75, hb * 0.72), (rb * 0.3, hb * 0.92), (0, hb)],
                                         segments: 8, origin: tip - tangent * (hb * 0.05), axis: tangent))

            case .pod:
                // A long thin upright seed-pod horn standing out of the old collar.
                let axis = normalize3(tangent + up * 0.8)
                collar(axis)
                let podLen = petalLen * 2.4
                let lean = normalize3(axis + out * 0.12)
                var pod = Mesh3()
                if pod.sweep(profile: .circle(radius: size * 0.0035, segments: 5),
                             along: [tip, tip + axis * (podLen * 0.5), tip + lean * podLen],
                             scales: [1, 0.9, 0.3]) {
                    groups.append(pod.smoothed(), color: poppyPod)
                }
            }
        }

        return Parts(stem: stem, foliage: foliage, blooms: groups.groups)
    }

    /// One thread segment of a dissected poppy leaf, presenting its face upward.
    private static func emitPoppyThread(_ mesh: inout Mesh3, from: Vec3, direction: Vec3, length: Double) {
        let up = Vec3(0, 1, 0)
        var face = up - direction * dot3(up, direction)
        face = len3(face) > 1e-6 ? normalize3(face) : Vec3(1, 0, 0)
        let across = normalize3(cross3(direction, face))
        BladeDescription.poppyThread.emit(into: &mesh, position: from + direction * (length * 0.46),
                                          xAxis: across, yAxis: direction, bentNormal: face,
                                          length: length, winding: .doubleSidedShell)
    }
}
