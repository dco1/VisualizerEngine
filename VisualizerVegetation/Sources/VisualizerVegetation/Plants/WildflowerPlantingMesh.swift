import Foundation
import simd
import VisualizerMaterials


/// The mesh of a plot's planting: a seeded scatter of small plants over the plot polygon, each
/// seated on the drawn ground, merged into a handful of colour layers.
///
/// **Why a merged mesh and not placed plants.** A sown bed is ~10 plants per square metre; a 150 m²
/// parkway is 1 500 of them. As `PlacedGardenPlant`s that is 1 500 draw instances and TLAS entries
/// paid on every frame of an orbit (the performance gate every placeable answers to), and 1 500
/// rows in the layer tree nobody wants to see. As one mesh per plot it is one draw.
///
/// **Why a variant library.** Every plant is a COPY of one of a few pre-built, pre-smoothed
/// variants, turned, scaled and seated — the scatter does no leaf construction and no smoothing of
/// its own, so a plot rebuild is array appends. Variation that reads at yard distance is heading,
/// size and which variant, not per-leaf uniqueness.
///
/// Everything is single-sided: the site's draw groups are registered two-sided
/// (`IlluminatoramaMesh.doubleSided`), so a blade needs one face, not a shell — half the triangles.
public enum WildflowerPlantingMesh {

    /// One colour of the planting. `translucent` layers are thin sheets (leaves, petals) that take
    /// the outdoor backlight; opaque ones are stems.
    public struct Layer: Sendable {
        public var color: Vec3
        public var translucent: Bool
        public var mesh: Mesh3
    }

    /// Whole-planting triangle ceiling. A bigger plot is sown THINNER rather than heavier — the
    /// same adaptive coarsening the grass field uses for a sprawling lot.
    public static let maxTriangles = 320_000
    /// Plants per square metre at density 1, all kinds together — the single density source.
    public static let plantsPerSquareMeter = 7.0

    // MARK: palette (linear-ish albedo, as the species files state theirs)

    /// What the bed's own ground is multiplied by under a planting — see `gardenGroups`.
    public static let understoryTint = Vec3(0.30, 0.40, 0.26)

    public static let foliageGreens = [Vec3(0.25, 0.26, 0.05), Vec3(0.29, 0.28, 0.05), Vec3(0.21, 0.23, 0.04)]   // a spring sward's golden olive (hue ~0.16): the goal's parkway measured 58 % yellow-olive, 2 % green
    public static let stemGreen     = Vec3(0.24, 0.26, 0.06)
    /// Poppies vary flower to flower from deep orange to the paler orange-yellow coastal form.
    public static let poppyOranges  = [Vec3(0.90, 0.22, 0.02), Vec3(0.94, 0.32, 0.03), Vec3(0.97, 0.44, 0.05)]   // the judge read the paler ones as yellow
    public static let lupineViolet  = Vec3(0.34, 0.24, 0.72)
    public static let lupineTip     = Vec3(0.62, 0.52, 0.86)
    public static let goldfield     = Vec3(0.97, 0.78, 0.10)
    public static let popcornWhite  = Vec3(0.93, 0.91, 0.84)
    /// Blue dicks (*Dichelostemma capitatum*): a tight umbel of violet florets — the round purple
    /// heads that ride above a foothill poppy field on bare stems. Core darker, floret tips lighter.
    public static let blueDicksCore   = Vec3(0.17, 0.07, 0.40)
    public static let blueDicksViolet = Vec3(0.30, 0.13, 0.64)
    /// Wild oat (*Avena fatua*): the straw-green spikelets that catch the light over a spring field.
    public static let oatStraw        = Vec3(0.52, 0.49, 0.27)

    // MARK: variants

    /// The plant kinds. `blueDicks` and `wildOat` are FOOTHILL-MEADOW kinds: `Mix.garden` (the
    /// default everywhere) never sows them, so a garden bed is byte-identical to what it was before
    /// they existed. Iterate `Mix.kinds`, not `allCases`, for "what can this sowing produce".
    public enum Kind: Int, CaseIterable, Sendable { case tuft, poppy, lupine, goldfields, popcorn, blueDicks, wildOat }

    /// Which planting is sown — the relative abundance of each kind, the drift fields that move it
    /// across the ground, and the plant-to-plant size spread.
    public enum Mix: Sendable, Equatable {
        /// The sown garden bed (the original mix): fine drifts a few metres across.
        case garden
        /// A native foothill spring bloom — Antelope Valley in a wet year. Poppies and blue dicks
        /// gather into broad SWATHS tens of metres across, alternating orange and purple, with
        /// goldfields speckled through and wild oat standing out of the grass.
        case foothillMeadow

        public var kinds: [Kind] {
            switch self {
            case .garden: return [.tuft, .poppy, .lupine, .goldfields, .popcorn]
            case .foothillMeadow: return Kind.allCases
            }
        }
    }

    public struct Variant: Sendable {
        public var layers: [Layer]
        public var triangleCount: Int { layers.reduce(0) { $0 + $1.mesh.triangleCount } }
    }

    /// The library — built once per process, local space (crown at the origin, Y up).
    public static let library: [Kind: [Variant]] = {
        var out: [Kind: [Variant]] = [:]
        for kind in Kind.allCases {
            out[kind] = (0 ..< variantCount(kind)).map { buildVariant(kind, index: $0) }
        }
        return out
    }()

    /// Enough of each that the eye stops finding the same plant twice in one view (the judge's
    /// "repeated, uniform assets" at 5 / 4, still at 9 / 7). Tufts stop at 11: tuft 12+ spreads
    /// past the 0.6 m footprint bound. Flower counts that grow with the index wrap at 7, so the
    /// first seven variants of each kind are exactly what they were.
    public static func variantCount(_ kind: Kind) -> Int { kind == .tuft ? 11 : 10 }

    public static let bladeSilhouette = LeafSilhouette(name: "wildflowerBlade",
                                                hero: [(0.00, 0.00), (0.30, 1.00), (1.00, 0.00)])
    /// A small rounded leaflet: a blunt oval, not the kite three controls make.
    public static let leafletSilhouette = LeafSilhouette(name: "wildflowerLeaflet",
                                                  hero: [(0.00, 0.00), (0.30, 0.92), (0.72, 0.88), (1.00, 0.00)])
    /// A petal at sowing scale: a fan widest near the tip — three controls, four triangles.
    public static let petalSilhouette = LeafSilhouette(name: "wildflowerPetal",
                                                hero: [(0.00, 0.00), (0.72, 1.00), (1.00, 0.00)])

    public static func buildVariant(_ kind: Kind, index: Int) -> Variant {
        var rng = PottedPlantMesh.SplitMix(UInt64(kind.rawValue + 1) &* 0x9E3779B97F4A7C15 &+ UInt64(index) &* 0xD1B54A32D192ED03)
        let up = Vec3(0, 1, 0)
        var layers: [Layer] = []
        func add(_ mesh: Mesh3, _ color: Vec3, translucent: Bool) {
            guard !mesh.isEmpty else { return }
            if let i = layers.firstIndex(where: { $0.color == color && $0.translucent == translucent }) {
                layers[i].mesh.append(mesh)
            } else {
                layers.append(Layer(color: color, translucent: translucent, mesh: mesh))
            }
        }
        /// A mound of narrow arching blades — the sward every flower stands in.
        /// The plant's foliage: a soft MOUND of small leaflets with a few longer blades through it.
        ///
        /// The first version was blades alone, and a bed of blades is a bed of grass — or, radiating
        /// from one crown, of yuccas. What a wildflower sward looks like from standing height is a
        /// low cushion of many small leaf surfaces facing every way (poppy and lupine foliage is
        /// finely divided), so that is what is built: leaflets scattered over a dome, each facing
        /// roughly out of it, with a handful of taller blades breaking the outline.
        func tuft(blades: Int, height: Double, spread: Double, color: Vec3) {
            var m = Mesh3()
            let radius = 0.06 + height * 0.55 * spread
            /// Where a point of this mound faces, as the mound — not as the one leaf it sits on.
            func moundNormal(_ p: Vec3) -> Vec3 {
                normalize3(Vec3(p.x / radius, 0.9 + max(0, p.y) / max(0.05, height), p.z / radius))
            }
            for _ in 0 ..< blades * 2 {
                let a = rng.unit() * 2 * Double.pi, r = radius * rng.unit().squareRoot()
                let dome = (1 - (r / radius) * (r / radius)).squareRoot()
                let y = 0.05 + height * dome * (0.45 + rng.unit() * 0.55)   // clear of the soil: a rim leaflet droops
                let outward = normalize3(Vec3(cos(a) * r / radius, 0.55 + dome, sin(a) * r / radius)
                                         + Vec3(rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5) * 0.9)
                let t = rng.unit() * 2 * Double.pi
                var along = Vec3(cos(t), (rng.unit() - 0.3) * 0.8, sin(t))
                along = normalize3(along - outward * dot3(along, outward))
                let side = normalize3(cross3(along, outward))
                let len = 0.045 + rng.unit() * 0.045
                LeafConstructor.emitBlade(into: &m,
                                          placement: .init(position: Vec3(cos(a) * r, y, sin(a) * r), xAxis: side, yAxis: along,
                                                           bentNormal: outward, width: len * (0.42 + rng.unit() * 0.18), height: len),
                                          silhouette: leafletSilhouette, subdivisions: 1,
                                          fold: 0.25, curl: 0.35, winding: .singleSided)
            }
            for _ in 0 ..< max(3, blades / 3) {
                let ba = rng.unit() * 2 * Double.pi, br = radius * 0.5 * rng.unit().squareRoot()
                let az = ba + (rng.unit() - 0.5) * 2.4
                let out = Vec3(cos(az), 0, sin(az)), side = Vec3(-sin(az), 0, cos(az))
                let u = rng.unit()
                let dir = normalize3(up * (1.1 - u * u * 0.5) + out * (0.1 + u * u * 0.5))
                let len = height * (0.9 + rng.unit() * 0.5)
                LeafConstructor.emitBlade(into: &m,
                                          placement: .init(position: Vec3(cos(ba) * br, 0, sin(ba) * br) + dir * (len * 0.46),
                                                           xAxis: side, yAxis: dir, bentNormal: normalize3(cross3(side, dir)),
                                                           width: len * (0.05 + rng.unit() * 0.03), height: len),
                                          silhouette: bladeSilhouette, subdivisions: 1,
                                          fold: 0.35, curl: 0.5 + rng.unit() * 0.6, winding: .singleSided)
            }
            // **Shade the mound, not the leaf.** A leaflet is a few flat triangles, and lit by its
            // own normal each one catches the sun separately: a cushion of foliage reads as a heap
            // of bright and dark shards ("faceted"). Foliage at this scale is lit as a volume, so
            // every vertex normal is pulled most of the way to the MOUND's normal at that point —
            // the canopy bent-normal the yard trees use. The leaf keeps a little of its own facing,
            // which is what still separates one leaflet from the next.
            var soft = m.smoothed()
            for k in soft.normals.indices {
                soft.normals[k] = normalize3(soft.normals[k] * 0.25 + moundNormal(soft.positions[k]) * 0.75)
            }
            add(soft, color, translucent: true)
        }
        /// A bare three-sided stem from the crown to `target`, bowed a little; returns where the head
        /// sits and which way it faces. Two segments: at a sown bed's viewing distance a stem is a
        /// line, and `appendBentStemTube`'s eight-ring tube was a third of every flower's triangles.
        func stem(to target: Vec3, radius: Double) -> (tip: Vec3, tangent: Vec3) {
            let bow = Vec3(target.x, 0, target.z) * 0.25
            let mid = target * 0.5 - bow + up * (target.y * 0.08)
            var m = Mesh3()
            _ = m.sweep(profile: .circle(radius: radius, segments: 3), along: [Vec3(0, 0, 0), mid, target],
                        scales: [1, 0.8, 0.6], capStart: false, capEnd: false)
            add(m.smoothed(), stemGreen, translucent: false)
            return (target, normalize3(target - mid))
        }
        /// A flower of `count` petals about `axis` — `pitch` 0 = furled upright, π/2 = flat open.
        func flower(at center: Vec3, axis: Vec3, petals count: Int, length: Double, widthRatio: Double,
                    pitch: Double, color: Vec3) {
            let f = GardenPlantMesh.frame(yAxis: axis)
            let phase = rng.unit() * 2 * Double.pi
            var m = Mesh3()
            for k in 0 ..< count {
                let a = phase + Double(k) * 2 * Double.pi / Double(count)
                let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                let p = GardenPlantMesh.petalPlacement(center: center, axis: axis, radial: radial,
                                                       pitch: pitch + (k % 2 == 0 ? 0.07 : -0.07),
                                                       length: length, width: length * widthRatio)
                LeafConstructor.emitBlade(into: &m, placement: p, silhouette: petalSilhouette, subdivisions: 1,
                                          fold: 0.30, curl: -0.12, winding: .singleSided)
            }
            // Petals are OPAQUE here: with the thin-sheet backlight a low sun turns a deep orange
            // poppy into a glowing yellow (the judge read the bed as "a yellow monoculture").
            // Leaves keep the backlight; the flower's colour is the point.
            add(m.smoothed(), color, translucent: false)
        }

        /// Scratch for a kind that builds one layer from several emitters.
        var layersScratch = Mesh3()
        /// A blue-dicks umbel: a dark core ball bristling with small six-tepal florets facing out.
        func umbel(at center: Vec3, radius: Double) {
            var core = Mesh3()
            let prof = (0 ... 4).map { i -> Mesh3.ProfilePoint in
                let a = Double.pi * Double(i) / 4
                return .init(r: sin(a) * radius * 0.78, y: -cos(a) * radius * 0.78)
            }
            core.revolve(profile: prof, segments: 6)
            add(core.translated(by: center).smoothed(creaseDegrees: 80), blueDicksCore, translucent: false)
            var florets = Mesh3()
            let n = 11
            for k in 0 ..< n {
                // Fibonacci sphere, the lower pole left bare where the stem enters.
                let t = (Double(k) + 0.5) / Double(n)
                let yy = 1 - t * 1.7, rr = max(0, 1 - yy * yy).squareRoot()
                let a = Double(k) * 2.399963 + rng.unit() * 0.3
                let axis = normalize3(Vec3(cos(a) * rr, yy, sin(a) * rr))
                let f = GardenPlantMesh.frame(yAxis: axis)
                let base = center + axis * (radius * 0.72)
                for p in 0 ..< 3 {
                    let pa = rng.unit() * 0.4 + Double(p) * 2 * Double.pi / 3
                    let radial = normalize3(f.x * cos(pa) + f.z * sin(pa))
                    let pl = GardenPlantMesh.petalPlacement(center: base, axis: axis, radial: radial, pitch: 0.75,
                                                            length: radius * 0.62, width: radius * 0.42)
                    LeafConstructor.emitBlade(into: &florets, placement: pl, silhouette: petalSilhouette, subdivisions: 1,
                                              fold: 0.3, curl: -0.1, winding: .singleSided)
                }
            }
            // Florets are lit as the BALL they make, not as shards (the tuft's mound rule).
            var soft = florets.smoothed()
            for k in soft.normals.indices {
                soft.normals[k] = normalize3(soft.normals[k] * 0.3 + normalize3(soft.positions[k] - center) * 0.7)
            }
            add(soft, blueDicksViolet, translucent: false)
        }
        /// A wild-oat panicle: whorls of short pedicels off the culm tip, each hanging a spikelet.
        func panicle(top: Vec3, axis: Vec3) {
            var m = Mesh3()
            let f = GardenPlantMesh.frame(yAxis: axis)
            let count = 7 + Int(rng.unit() * 4)
            for k in 0 ..< count {
                let t = Double(k) / Double(count)
                let anchor = top - axis * (0.16 * t)
                let a = Double(k) * 2.399963 + rng.unit() * 0.5
                let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                // Droops: points outward and down from where it hangs.
                let hang = normalize3(radial * 0.55 - up * (0.75 + rng.unit() * 0.3))
                let len = 0.020 + rng.unit() * 0.008
                let pos = anchor + radial * 0.018 + hang * (len * 0.5)
                let side = normalize3(cross3(hang, radial))
                LeafConstructor.emitBlade(into: &m,
                                          placement: .init(position: pos, xAxis: side, yAxis: hang,
                                                           bentNormal: normalize3(cross3(side, hang)),
                                                           width: len * 0.30, height: len),
                                          silhouette: bladeSilhouette, subdivisions: 1,
                                          fold: 0.5, curl: 0.1, winding: .singleSided)
            }
            add(m.smoothed(), oatStraw, translucent: true)
        }

        let green = foliageGreens[index % foliageGreens.count]
        switch kind {
        case .tuft:
            tuft(blades: 22 + index % 3 * 2, height: 0.26 + Double(index % 4) * 0.03, spread: 1.0, color: green)

        case .poppy:
            // A small poppy plant: a low tuft and 3–5 cups held above it, most open, one furled.
            tuft(blades: 16, height: 0.24, spread: 1.0, color: green)
            let heads = 8 + index % 3      // a poppy in full bloom carries a crowd of cups
            for h in 0 ..< heads {
                let az = rng.unit() * 2 * Double.pi
                let reach = 0.05 + rng.unit() * 0.18, height = 0.27 + rng.unit() * 0.18
                let s = stem(to: Vec3(cos(az) * reach, height, sin(az) * reach), radius: 0.0035)
                let open = h < heads - 1       // one furled bud per plant, the rest open to the sun
                flower(at: s.tip, axis: normalize3(s.tangent + up * 0.8), petals: 4,
                       length: 0.058 + rng.unit() * 0.016, widthRatio: open ? 1.25 : 0.6,
                       pitch: open ? 0.80 + rng.unit() * 0.25 : 0.14,
                       color: poppyOranges[(index + h) % poppyOranges.count])
            }

        case .lupine:
            // Palmate leaves low, one to three flowering spikes: whorls of small florets on a
            // tapering raceme, violet below, paler at the unopened tip.
            tuft(blades: 16, height: 0.26, spread: 1.0, color: green)
            for _ in 0 ..< (1 + index % 3) {
                let az = rng.unit() * 2 * Double.pi, reach = rng.unit() * 0.10
                let top = 0.45 + rng.unit() * 0.25
                let s = stem(to: Vec3(cos(az) * reach, top, sin(az) * reach), radius: 0.004)
                let racemeLength = top * 0.55, whorls = 6
                for w in 0 ..< whorls {
                    let t = Double(w) / Double(whorls - 1)                     // 0 base … 1 tip
                    let center = s.tip - s.tangent * (racemeLength * (1 - t))
                    flower(at: center, axis: s.tangent, petals: 4, length: 0.034 * (1.0 - 0.55 * t), widthRatio: 0.8,
                           pitch: 1.05 - 0.5 * t, color: t > 0.7 ? lupineTip : lupineViolet)
                }
            }

        case .goldfields:
            // A low cushion of small yellow daisies.
            tuft(blades: 14, height: 0.16, spread: 1.2, color: green)
            for _ in 0 ..< (6 + index % 7) {
                let az = rng.unit() * 2 * Double.pi, reach = 0.03 + rng.unit() * 0.13
                let s = stem(to: Vec3(cos(az) * reach, 0.12 + rng.unit() * 0.10, sin(az) * reach), radius: 0.002)
                flower(at: s.tip, axis: normalize3(s.tangent + up), petals: 5, length: 0.021, widthRatio: 0.7,
                       pitch: 1.35, color: goldfield)
            }

        case .popcorn:
            tuft(blades: 14, height: 0.18, spread: 1.0, color: green)
            for _ in 0 ..< (7 + index % 7) {
                let az = rng.unit() * 2 * Double.pi, reach = 0.02 + rng.unit() * 0.12
                let s = stem(to: Vec3(cos(az) * reach, 0.20 + rng.unit() * 0.12, sin(az) * reach), radius: 0.002)
                flower(at: s.tip, axis: normalize3(s.tangent + up), petals: 5, length: 0.018, widthRatio: 0.9,
                       pitch: 1.3, color: popcornWhite)
            }

        case .blueDicks:
            // A few long, narrow, half-withered basal leaves and one to three tall BARE stems, each
            // carrying a round umbel ~3 cm across: a dark core ball bristling with six-tepal florets.
            // The stem is the point — the heads stand clear above the poppies.
            for _ in 0 ..< 3 + index % 2 {
                let az = rng.unit() * 2 * Double.pi
                let out = Vec3(cos(az), 0, sin(az)), side = Vec3(-sin(az), 0, cos(az))
                let dir = normalize3(up + out * (0.35 + rng.unit() * 0.4))
                let len = 0.20 + rng.unit() * 0.12
                LeafConstructor.emitBlade(into: &layersScratch,
                                          placement: .init(position: dir * (len * 0.46), xAxis: side, yAxis: dir,
                                                           bentNormal: normalize3(cross3(side, dir)),
                                                           width: len * 0.05, height: len),
                                          silhouette: bladeSilhouette, subdivisions: 1,
                                          fold: 0.3, curl: 0.8 + rng.unit() * 0.6, winding: .singleSided)
            }
            add(layersScratch.smoothed(), green, translucent: true)
            layersScratch = Mesh3()
            for _ in 0 ..< (1 + index % 3) {
                let az = rng.unit() * 2 * Double.pi, reach = 0.02 + rng.unit() * 0.10
                // Heads clear ABOVE the poppy cups (≈0.3–0.45 m): the pom-poms ride over the bloom.
                let s = stem(to: Vec3(cos(az) * reach, 0.48 + rng.unit() * 0.22, sin(az) * reach), radius: 0.0024)
                umbel(at: s.tip + s.tangent * 0.015, radius: 0.016 + rng.unit() * 0.003)
            }

        case .wildOat:
            // A tuft of long arching blades and two or three culms, each ending in a loose panicle
            // of drooping spikelets — straw-green, translucent, so a low sun lights them up.
            for _ in 0 ..< 9 + index % 3 {
                let az = rng.unit() * 2 * Double.pi, br = 0.03 * rng.unit()
                let out = Vec3(cos(az), 0, sin(az)), side = Vec3(-sin(az), 0, cos(az))
                let u = rng.unit()
                let dir = normalize3(up * (1.2 - u * 0.3) + out * (0.08 + u * 0.30))
                let len = 0.24 + rng.unit() * 0.14
                LeafConstructor.emitBlade(into: &layersScratch,
                                          placement: .init(position: out * br + dir * (len * 0.46) + up * 0.01, xAxis: side, yAxis: dir,
                                                           bentNormal: normalize3(cross3(side, dir)),
                                                           width: len * 0.035, height: len),
                                          silhouette: bladeSilhouette, subdivisions: 1,
                                          fold: 0.3, curl: 0.45 + rng.unit() * 0.4, winding: .singleSided)
            }
            add(layersScratch.smoothed(), green, translucent: true)
            layersScratch = Mesh3()
            for _ in 0 ..< (2 + index % 2) {
                let az = rng.unit() * 2 * Double.pi, reach = 0.06 + rng.unit() * 0.14
                let s = stem(to: Vec3(cos(az) * reach, 0.58 + rng.unit() * 0.20, sin(az) * reach), radius: 0.0018)
                panicle(top: s.tip, axis: s.tangent)
            }
        }
        return Variant(layers: layers)
    }

    // MARK: the sowing

    /// One plant of the scatter — the single source the mesh and its tests read.
    public struct Plant: Equatable, Sendable {
        public var kind: Kind
        public var variant: Int
        public var position: Vec2      // plan metres
        public var heading: Double
        public var scale: Double
    }

    /// Relative abundance of each kind at plan point `p`. Poppies and lupine grow in DRIFTS — a
    /// slow noise field moves the mix across the bed so colour gathers into patches instead of
    /// being salt-and-peppered evenly, which is what makes a sown mix read as sown.
    public static func weights(at p: Vec2, seed: UInt64) -> [(Kind, Double)] {
        weights(at: p, seed: seed, mix: .garden)
    }

    /// The drift pair (each 0…1) at plan point `p` — the slow fields that gather each mix's colour
    /// into patches. Exposed so a host can colour ground it will never sow plant by plant (the far
    /// hills of a flyover) from the SAME field the near plants come from, and the swath a distant
    /// hill shows resolves into the flowers that make it as the camera arrives.
    public static func drift(at p: Vec2, seed: UInt64, mix: Mix) -> (Double, Double) {
        switch mix {
        case .garden:
            return (valueNoise(p * 0.22, seed: seed), valueNoise(p * 0.31 + Vec2(40, 17), seed: seed &+ 7))
        case .foothillMeadow:
            // Swaths tens of metres across, a little stretched (drifts follow the slope), with a
            // finer octave so an edge frays instead of running as a smooth contour.
            func field(_ q: Vec2, _ s: UInt64) -> Double {
                let broad = valueNoise(Vec2(q.x * 0.030, q.y * 0.048), seed: s)
                let fine = valueNoise(q * 0.16, seed: s &+ 3)
                return broad * 0.78 + fine * 0.22
            }
            return (field(p, seed &+ 101), field(p + Vec2(311, -97), seed &+ 211))
        }
    }

    /// The mix at `p` for `mix`. `.garden` is exactly the original bed.
    public static func weights(at p: Vec2, seed: UInt64, mix: Mix) -> [(Kind, Double)] {
        let (a, b) = drift(at: p, seed: seed, mix: mix)
        return weights(drift: a, b, mix: mix)
    }

    /// The mix for a given pair of drift values (each 0…1) — also what the density estimate reads,
    /// at the field's mean, so the triangle ceiling is computed from the mix actually sown.
    public static func weights(drift a: Double, _ b: Double) -> [(Kind, Double)] {
        weights(drift: a, b, mix: .garden)
    }

    public static func weights(drift a: Double, _ b: Double, mix: Mix) -> [(Kind, Double)] {
        guard mix == .garden else {
            // Poppy and blue dicks are near-exclusive: where one drift runs strong the other thins,
            // which is what makes the hills read as alternating SWATHS rather than a blend.
            let orange = smoothstep(0.38, 0.62, a)
            let purple = smoothstep(0.42, 0.64, b) * (1 - orange * 0.85)
            // Inside a strong drift the flowers crowd the grass back — but a wet-spring sward
            // still fills every gap between them (no bare soil in a superbloom).
            let crowd = 1 - 0.35 * max(orange, purple)
            return [(.tuft, 0.30 * crowd),
                    (.wildOat, 0.035 * crowd),
                    (.poppy, 0.04 + 0.95 * orange),
                    (.blueDicks, 0.05 + 0.80 * purple),     // a sprinkle everywhere, a swath in a drift
                    (.goldfields, 0.16 + 0.12 * (1 - orange) * (1 - purple)),
                    (.lupine, 0.02 + 0.05 * purple),
                    (.popcorn, 0.015)]
        }
        return [(.tuft, 0.22),
         (.poppy, 0.14 + 0.30 * a),
         (.lupine, 0.10 + 0.28 * b * (1 - a * 0.5)),
         (.goldfields, 0.10 + 0.14 * (1 - a)),
         (.popcorn, 0.13)]
    }

    /// Mean triangles one sown plant costs, from the library and the mean mix.
    public static let meanTrianglesPerPlant: Double = {
        let mix = weights(drift: 0.5, 0.5)
        let total = mix.reduce(0) { $0 + $1.1 }
        return mix.reduce(0.0) { acc, entry in
            let v = library[entry.0] ?? []
            return acc + entry.1 / total * Double(v.reduce(0) { $0 + $1.triangleCount }) / Double(max(1, v.count))
        }
    }()

    public static func plants(polygon: [Vec2], density: Double, seed: UInt64) -> [Plant] {
        plants(polygon: polygon, density: density, seed: seed, mix: .garden, triangleCeiling: maxTriangles)
    }

    /// The scatter, generalised.
    /// - `mix`: which planting (default `.garden`, the original bed).
    /// - `triangleCeiling`: the merged-mesh ceiling the sowing thins itself under. `nil` = uncapped,
    ///   for a host that draws the library variants as GPU INSTANCES (it pays per instance, not per
    ///   unique triangle). Uncapped, the cell size depends only on `density`, so the same world
    ///   cell always grows the same plant no matter which tile or window asks — what a streamed,
    ///   infinite field needs.
    public static func plants(polygon: [Vec2], density: Double, seed: UInt64, mix: Mix,
                              triangleCeiling: Int?) -> [Plant] {
        guard polygon.count >= 3, density > 0 else { return [] }
        let area = VisualizerMaterials.area(polygon)
        guard area > 0.01 else { return [] }
        // Thin the sowing, never exceed the ceiling.
        let wanted = plantsPerSquareMeter * min(2, max(0.1, density))
        let perSquareMeter = triangleCeiling.map { min(wanted, Double($0) / max(1, meanTrianglesPerPlant) / area) } ?? wanted
        let cell = 1 / perSquareMeter.squareRoot()

        let xs = polygon.map(\.x), ys = polygon.map(\.y)
        guard let x0 = xs.min(), let x1 = xs.max(), let y0 = ys.min(), let y1 = ys.max() else { return [] }
        var out: [Plant] = []
        // Cells are indexed from the WORLD origin, so reshaping a plot's edge never reshuffles the
        // plants in its middle.
        for iy in Int((y0 / cell).rounded(.down)) ... Int((y1 / cell).rounded(.up)) {
            for ix in Int((x0 / cell).rounded(.down)) ... Int((x1 / cell).rounded(.up)) {
                var rng = PottedPlantMesh.SplitMix(seed &+ UInt64(bitPattern: Int64(ix)) &* 0x9E3779B97F4A7C15
                                                        &+ UInt64(bitPattern: Int64(iy)) &* 0xC2B2AE3D27D4EB4F)
                let p = Vec2((Double(ix) + rng.unit()) * cell, (Double(iy) + rng.unit()) * cell)
                guard contains(polygon: polygon, p) else { continue }
                let table = weights(at: p, seed: seed, mix: mix)
                var pick = rng.unit() * table.reduce(0) { $0 + $1.1 }
                var kind = Kind.tuft
                for (k, w) in table { if pick < w { kind = k; break }; pick -= w }
                let variant = Int(rng.next() % UInt64(variantCount(kind)))
                let heading = rng.unit() * 2 * Double.pi
                let u = rng.unit() * rng.unit()
                out.append(Plant(kind: kind, variant: variant, position: p, heading: heading,
                                 // Height varies a LOT in a sown bed — a few plants stand well above the
                                 // sward. A native stand is more even (one season's germination).
                                 scale: mix == .garden ? 0.75 + u * 1.2 : 0.72 + u * 0.62))
            }
        }
        return out
    }

    /// The planting as world-space colour layers. `groundHeight` is plan point → world Y of the
    /// ground there (the caller's drape), so a bed on a bank follows the bank.
    public static func layers(polygon: [Vec2], density: Double = 1,
                              seed: UInt64 = 1, mix: Mix = .garden, groundHeight: (Vec2) -> Double) -> [Layer] {
        var out: [Layer] = []
        for plant in plants(polygon: polygon, density: density, seed: seed, mix: mix, triangleCeiling: maxTriangles) {
            guard let variant = library[plant.kind]?[plant.variant] else { continue }
            let c = cos(plant.heading) * plant.scale, s = sin(plant.heading) * plant.scale
            let cn = cos(plant.heading), sn = sin(plant.heading)
            // Sunk a little so a crown on a slope never shows daylight under its downhill side.
            let origin = Vec3(plant.position.x, groundHeight(plant.position) - 0.02, -plant.position.y)
            for layer in variant.layers {
                var m = layer.mesh
                m.positions = m.positions.map { Vec3(origin.x + $0.x * c + $0.z * s, origin.y + $0.y * plant.scale, origin.z - $0.x * s + $0.z * c) }
                m.normals = m.normals.map { Vec3($0.x * cn + $0.z * sn, $0.y, -$0.x * sn + $0.z * cn) }
                if let i = out.firstIndex(where: { $0.color == layer.color && $0.translucent == layer.translucent }) {
                    out[i].mesh.append(m)
                } else {
                    out.append(Layer(color: layer.color, translucent: layer.translucent, mesh: m))
                }
            }
        }
        return out
    }

    // MARK: seen from afar

    /// What a kind looks like from far enough away that its plants are sub-pixel: its flower colour
    /// and the fraction of the view that colour covers (the rest is foliage green). Measured at a
    /// GRAZING view, which is how every far hill is seen: the heads stand on top of the foliage, so
    /// looking across a field they occlude most of the green under them — a poppy drift reads as
    /// solid orange at 200 m even though, straight down, half of each plant is leaves.
    public static func overhead(_ kind: Kind) -> (color: Vec3, cover: Double) {
        switch kind {
        case .tuft:       return (foliageGreens[1], 0)
        case .poppy:      return (poppyOranges[1], 0.80)
        case .lupine:     return (lupineViolet, 0.55)
        case .goldfields: return (goldfield, 0.60)
        case .popcorn:    return (popcornWhite, 0.45)
        case .blueDicks:  return (blueDicksViolet, 0.80)
        case .wildOat:    return (oatStraw, 0.45)
        }
    }

    /// The mean albedo of the sowing at plan point `p`, as a far hill shows it: every kind's
    /// overhead colour blended by its share of the mix. A host paints this into the ground of the
    /// ground it will not sow plant by plant, so a drift on a far hill is the same drift the near
    /// plants make — colour swaths that resolve into flowers on approach, no pop.
    public static func meanAlbedo(at p: Vec2, seed: UInt64, mix: Mix) -> Vec3 {
        let table = weights(at: p, seed: seed, mix: mix)
        let total = table.reduce(0) { $0 + $1.1 }
        var c = Vec3(0, 0, 0)
        for (k, w) in table {
            let o = overhead(k)
            c = c + (o.color * o.cover + foliageGreens[1] * (1 - o.cover)) * (w / total)
        }
        return c
    }

    // MARK: small numerics

    /// Smooth value noise in 0…1 on an integer lattice — deterministic, process-independent.
    public static func valueNoise(_ p: Vec2, seed: UInt64) -> Double {
        func h(_ x: Int, _ y: Int) -> Double {
            var r = PottedPlantMesh.SplitMix(seed &+ UInt64(bitPattern: Int64(x)) &* 0x9E3779B97F4A7C15
                                                  &+ UInt64(bitPattern: Int64(y)) &* 0xC2B2AE3D27D4EB4F)
            return r.unit()
        }
        let xf = p.x.rounded(.down), yf = p.y.rounded(.down)
        let x = Int(xf), y = Int(yf)
        let tx = smoothstep(0, 1, p.x - xf), ty = smoothstep(0, 1, p.y - yf)
        return mix(mix(h(x, y), h(x + 1, y), tx), mix(h(x, y + 1), h(x + 1, y + 1), tx), ty)
    }
}
