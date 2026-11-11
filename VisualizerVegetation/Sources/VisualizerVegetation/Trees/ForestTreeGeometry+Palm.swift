import Foundation
import VisualizerMaterials
import simd

// A PALM — the factory's third growth mode. A broadleaf is arms off a trunk top with a leaf fill; a
// conifer is whorls of sprigged branches; a palm is neither: ONE unbranched stem, and a single
// crown of very large leaves that all leave the same growing point. Old leaves are pushed down and
// out by new ones, so the crown is a sphere of fronds ordered by age: a spear standing at the top,
// a horizontal ring, and the oldest hanging below it.
//
// Three leaf forms cover the palms and palm-like plants a street is planted with:
//
// * **feather** (king, queen, date) — a long arching midrib with a leaflet either side all along it;
// * **fan** (Mexican / California fan) — a stiff stalk carrying one pleated round blade whose
//   outer third splits into drooping fingers;
// * **sword** (yucca, dracaena) — no stalk at all: a rosette of stiff pointed blades, and a stem
//   that forks into a few heads.
//
// The unit of level of detail is the FROND (one leaf card): thinning a crown removes whole fronds,
// never half the leaflets of one. Lengths in `PalmLook` are metres at `referenceHeight`; a young
// palm carries nearly full-sized leaves on a short stem, so they shrink far slower than the height.
extension ForestTreeGeometry {

    /// How a palm is grown. Lengths are metres for a palm `referenceHeight` tall.
    public struct PalmLook: Sendable {
        public enum Frond: Sendable { case feather, fan, sword }
        public var frond: Frond
        public var referenceHeight: Float
        /// Stem radius at mid height, and how much wider the foot is.
        public var trunkRadius: Float
        public var baseFlare: Float
        public var frondCount: Int
        /// Feather: the midrib. Fan: stalk + blade. Sword: one blade.
        public var frondLength: Float
        /// How far a frond arcs under its own weight (0 stiff … 1 weeping).
        public var droop: Float
        /// Elevation (radians) of the OLDEST living frond — below zero hangs under the crown.
        public var lowestElevation: Float
        /// Dead fronds left hanging against the stem, as a fraction of `frondCount`, and the length
        /// of stem (m) they cover.
        public var skirt: Float = 0
        public var skirtLength: Float = 0
        /// A smooth green sheath between the stem and the crown (m); 0 for none.
        public var crownshaft: Float = 0
        /// Sword rosettes only: the most heads one plant carries.
        public var heads: Int = 1
        public var leafTip: SIMD3<Float>           // linear albedo at the leaf tip
        public var dead: SIMD3<Float>              // linear albedo of a dead frond
        public var shaft: SIMD3<Float> = .zero     // linear albedo of the crownshaft
    }

    /// Bake a palm into the soup. `site.height` is the whole plant, ground to the top of the crown.
    static func emitPalm(_ s: inout Soup, world: simd_float4x4, look: TreeLook, palm p: PalmLook, site: TreeSite) {
        var rng = ForestRNG(seed: site.seed &+ 0x9A1A_0001)
        let H = max(0.6, site.height)
        // Leaves and stem gauge grow far slower than the stem's height.
        let k = max(0.45, min(1.2, pow(H / p.referenceHeight, 0.4)))
        let frondL = p.frondLength * k
        let trunkR = p.trunkRadius * k * (0.9 + 0.2 * rng.unit())

        if p.frond == .sword {
            emitSwordPalm(&s, world: world, look: look, palm: p, H: H, blade: frondL, trunkR: trunkR, rng: &rng)
            return
        }

        // The crown stands above the stem top by the rise of its upright fronds.
        let rise = frondL * (p.frond == .fan ? 0.78 : 0.86)
        let stemTop = max(H * 0.18, H - rise)
        // A stem is never ruler straight: one slow bow, bigger on a tall slender palm.
        let bow = stemTop * (0.012 + 0.03 * rng.unit()) * (p.trunkRadius < 0.2 ? 1.6 : 0.7)
        let bowAz = rng.unit() * 2 * .pi
        func axis(_ y: Float) -> SIMD3<Float> {
            let t = y / max(1e-3, stemTop)
            let off = bow * sin(t * .pi * 0.5) * t
            return SIMD3(cos(bowAz) * off, y, sin(bowAz) * off)
        }
        let shaftLen = min(p.crownshaft * k, stemTop * 0.4)
        let woodTop = stemTop - shaftLen
        let stations = 12
        var path: [SIMD3<Float>] = [], radii: [Float] = [], cols: [SIMD3<Float>] = []
        for i in 0 ... stations {
            let t = Float(i) / Float(stations)
            path.append(axis(woodTop * t))
            // Foot flare over the first tenth, then a slow taper.
            let flare = 1 + (p.baseFlare - 1) * pow(max(0, 1 - t / 0.12), 2)
            radii.append(trunkR * flare * (1.0 - 0.14 * t))
            // Leaf-scar rings: alternate stations a shade apart.
            cols.append(look.bark * (i % 2 == 0 ? 1.0 : 0.86))
        }
        s.debugClass = 1
        palmTube(&s, world: world, path: path, radii: radii, colors: cols, sides: 10)
        if shaftLen > 0.02 {
            let n = 4
            let sp = (0 ... n).map { axis(woodTop + shaftLen * Float($0) / Float(n)) }
            let sr = (0 ... n).map { i -> Float in
                let t = Float(i) / Float(n)
                return trunkR * 0.86 * (1.22 - 0.30 * t * t)        // swollen at the foot, narrowing into the crown
            }
            palmTube(&s, world: world, path: sp, radii: sr, colors: Array(repeating: p.shaft, count: n + 1), sides: 10)
        }
        s.debugClass = 0

        let hub = axis(stemTop)
        let count = max(5, Int((Float(p.frondCount) * max(0.5, min(1.3, site.foliageScale / 0.30))).rounded()))
        for i in 0 ..< count {
            // Youngest first: a spear at the top, then outward and down with age.
            let t = (Float(i) + 0.5) / Float(count)
            let elev = mix(1.38, p.lowestElevation, pow(t, 0.85)) + (rng.unit() - 0.5) * 0.14
            let az = Float(i) * 2.399963 + (rng.unit() - 0.5) * 0.35
            let len = frondL * (0.82 + 0.26 * rng.unit()) * (0.80 + 0.20 * sin(.pi * min(1, t * 1.15)))
            // Old fronds yellow before they die.
            let tone = mix(1.0, 0.78, t) * (0.9 + 0.2 * rng.unit())
            let tip = mixClamped(p.leafTip, p.dead, max(0, t - 0.82) * 2.2)
            emitFrond(&s, world: world, kind: p.frond, base: hub + SIMD3(0, shaftLen > 0 ? 0 : -trunkR * 0.3 * t, 0),
                      azimuth: az, elevation: elev, length: len, droop: p.droop,
                      root: look.leaf * tone, tip: tip * tone, k: k, rng: &rng)
        }
        // The skirt: dead fronds folded down against the stem.
        let dead = Int((Float(count) * p.skirt).rounded())
        for i in 0 ..< dead {
            let u = (Float(i) + rng.unit()) / Float(max(1, dead))
            let y = stemTop - shaftLen - u * p.skirtLength * k
            guard y > H * 0.1 else { continue }
            let tone = 0.75 + 0.4 * rng.unit()
            emitFrond(&s, world: world, kind: p.frond, base: axis(y), azimuth: rng.unit() * 2 * .pi,
                      elevation: -1.25 - 0.2 * rng.unit(), length: frondL * (0.55 + 0.25 * rng.unit()), droop: 0.15,
                      root: p.dead * tone * 0.8, tip: p.dead * tone, k: k, rng: &rng)
        }
    }

    // MARK: - One frond (a single leaf card)

    /// A frond leaving `base` (local space) toward `azimuth` at `elevation` above the horizon.
    private static func emitFrond(_ s: inout Soup, world: simd_float4x4, kind: PalmLook.Frond, base: SIMD3<Float>,
                                  azimuth: Float, elevation: Float, length L: Float, droop: Float,
                                  root: SIMD3<Float>, tip: SIMD3<Float>, k: Float, rng: inout ForestRNG) {
        let prevCard = s.cardMark, prevMark = s.foliageMark, prevDbg = s.debugClass
        s.cardMark = s.nextCardID; s.nextCardID &+= 1
        s.foliageMark = 1; s.debugClass = 6
        defer { s.cardMark = prevCard; s.foliageMark = prevMark; s.debugClass = prevDbg }

        let radial = SIMD3<Float>(cos(azimuth), 0, sin(azimuth))
        let side = SIMD3<Float>(-sin(azimuth), 0, cos(azimuth))
        let up = SIMD3<Float>(0, 1, 0)
        // The midrib: it leaves at `elevation` and gravity turns it down along its length.
        let stalkFrac: Float = kind == .fan ? 0.55 : 1
        let n = kind == .fan ? 4 : 9
        let bend = droop * (1.5 + 0.5 * cos(elevation)) * (kind == .fan ? 0.5 : 1)
        var spine: [SIMD3<Float>] = [base], tangents: [SIMD3<Float>] = []
        for j in 0 ..< n {
            let t = (Float(j) + 0.5) / Float(n)
            let pitch = elevation - bend * t * t
            let d = radial * cos(pitch) + up * sin(pitch)
            tangents.append(d)
            spine.append(spine[j] + d * (L * stalkFrac / Float(n)))
        }
        tangents.append(tangents[n - 1])
        // The frond's own "top side": perpendicular to the midrib, toward the sky.
        func topSide(_ tan: SIMD3<Float>) -> SIMD3<Float> { safeNormalize(simd_cross(side, tan), fallback: up) }
        let stalkCol = mixClamped(root, tip, 0.5)
        let stalkW = max(0.012, (kind == .fan ? 0.022 : 0.03) * k)
        for j in 0 ..< n {
            let w0 = stalkW * (1 - 0.6 * Float(j) / Float(n)), w1 = stalkW * (1 - 0.6 * Float(j + 1) / Float(n))
            let a = spine[j], b = spine[j + 1]
            let h = topSide(tangents[j])
            palmQuad(&s, world, a - side * w0, a + side * w0, b + side * w1, b - side * w1, hint: h, stalkCol, stalkCol)
        }

        switch kind {
        case .feather:
            let pairs = max(10, Int((22 * (L / (2.6 * k))).rounded()))
            let leafL = L * 0.21
            for i in 0 ..< pairs {
                let t = 0.10 + 0.90 * (Float(i) + 0.5) / Float(pairs)
                let x = t * Float(n), j = min(n - 1, Int(x))
                let at = spine[j] + (spine[j + 1] - spine[j]) * (x - Float(j))
                let tan = tangents[j], top = topSide(tan)
                // Longest a third of the way out, short at the foot and the tip.
                let len = leafL * pow(max(0.05, sin(.pi * (0.10 + 0.86 * t))), 0.7) * (0.9 + 0.2 * rng.unit())
                let w = max(0.012, 0.036 * k * (0.6 + 0.6 * sin(.pi * t)))
                let c0 = mixClamped(root, tip, 0.25 + 0.5 * t), c1 = mixClamped(root, tip, 0.65 + 0.35 * t)
                for sd: Float in [-1, 1] {
                    // Swept forward, raised in a V off the midrib, then let hang.
                    let sweep = 0.45 + 0.35 * t + (rng.unit() - 0.5) * 0.15
                    let vee = 0.38 * (1 - droop * 0.7)
                    let out = safeNormalize(side * (sd * cos(sweep)) + tan * sin(sweep) + top * vee)
                    let mid = at + out * (len * 0.55) - up * (len * 0.10 * droop)
                    let end = at + out * len - up * (len * (0.18 + 0.55 * droop))
                    let wd = safeNormalize(simd_cross(out, top), fallback: tan) * w
                    palmQuad(&s, world, at - wd * 0.6, at + wd * 0.6, mid + wd, mid - wd, hint: top, c0, c1)
                    palmTri(&s, world, mid - wd, mid + wd, end, hint: top, c1, c1, tip)
                }
            }
        case .fan:
            // One pleated round blade on the end of the stalk, its outer third split into fingers.
            let hubP = spine[n], fwd = tangents[n - 1], top = topSide(fwd)
            let R = L * (1 - stalkFrac) * (0.9 + 0.2 * rng.unit())
            let segs = 20
            let spread: Float = 1.95                       // half-angle of the blade (radians)
            let inner = R * 0.62
            for q in 0 ..< segs {
                let a0 = -spread + 2 * spread * Float(q) / Float(segs)
                let a1 = -spread + 2 * spread * Float(q + 1) / Float(segs)
                let am = (a0 + a1) * 0.5
                func dir(_ a: Float) -> SIMD3<Float> { fwd * cos(a) + side * sin(a) }
                // Pleats: ridge up on one edge, valley on the other. Side segments are shorter.
                let reach = 0.72 + 0.28 * cos(am * 0.8)
                let pleat = R * 0.035
                let pa = hubP + dir(a0) * (inner * reach) + top * pleat
                let pb = hubP + dir(a1) * (inner * reach) - top * pleat
                let sag = R * reach * (0.10 + 0.45 * droop) * (0.7 + 0.6 * rng.unit())
                let tipP = hubP + dir(am) * (R * reach) - up * sag
                let c1 = mixClamped(root, tip, 0.6) * (q % 2 == 0 ? 1.0 : 0.88)
                palmTri(&s, world, hubP, pa, pb, hint: top, root, c1, c1)
                palmTri(&s, world, pa, tipP, pb, hint: top, c1, tip, c1)
            }
        case .sword:
            break
        }
    }

    // MARK: - Sword rosettes (yucca)

    private static func emitSwordPalm(_ s: inout Soup, world: simd_float4x4, look: TreeLook, palm p: PalmLook,
                                      H: Float, blade: Float, trunkR: Float, rng: inout ForestRNG) {
        // Heads: a short plant is one rosette; a taller one has forked.
        let heads = max(1, min(p.heads, Int(1 + (H / p.referenceHeight) * Float(p.heads - 1) * (0.6 + 0.6 * rng.unit()))))
        let headRise = blade * 0.55
        let top = max(blade * 0.3, H - headRise)
        var stems: [[SIMD3<Float>]] = []
        let forkY = top * (0.35 + 0.25 * rng.unit())
        for h in 0 ..< heads {
            if h == 0 {
                stems.append((0 ... 6).map { SIMD3<Float>(0, top * Float($0) / 6, 0) })
            } else {
                // An arm leaves the stem, runs out, and turns up.
                let az = Float(h) * 2.399963 + rng.unit()
                let reach = top * (0.18 + 0.14 * rng.unit()) + blade * 0.35
                let endY = top * (0.72 + 0.22 * rng.unit())
                let y0 = forkY * (0.8 + 0.4 * rng.unit())
                stems.append((0 ... 6).map { i -> SIMD3<Float> in
                    let t = Float(i) / 6
                    let r = reach * sin(t * .pi * 0.5)
                    return SIMD3(cos(az) * r, y0 + (endY - y0) * (1 - cos(t * .pi * 0.5)), sin(az) * r)
                })
            }
        }
        s.debugClass = 1
        for (h, path) in stems.enumerated() {
            let r0 = trunkR * (h == 0 ? 1 : 0.72)
            let radii = (0 ... 6).map { i -> Float in
                let t = Float(i) / 6
                return r0 * (h == 0 ? 1 + (p.baseFlare - 1) * pow(1 - t, 3) : 1) * (1 - 0.12 * t)
            }
            palmTube(&s, world: world, path: path, radii: radii,
                     colors: (0 ... 6).map { look.bark * ($0 % 2 == 0 ? 1.0 : 0.88) }, sides: 8)
        }
        s.debugClass = 0

        let up = SIMD3<Float>(0, 1, 0)
        for path in stems {
            let hub = path[path.count - 1]
            let count = max(24, p.frondCount)
            // Cards of a dozen blades: thinning a rosette removes sectors, not slivers.
            let perCard = 12
            let prevCard = s.cardMark, prevMark = s.foliageMark, prevDbg = s.debugClass
            s.foliageMark = 1; s.debugClass = 6
            for i in 0 ..< count {
                if i % perCard == 0 { s.cardMark = s.nextCardID; s.nextCardID &+= 1 }
                let t = (Float(i) + 0.5) / Float(count)
                let elev = mix(1.45, p.lowestElevation, pow(t, 0.9)) + (rng.unit() - 0.5) * 0.12
                let az = Float(i) * 2.399963
                let radial = SIMD3<Float>(cos(az), 0, sin(az)), side = SIMD3<Float>(-sin(az), 0, cos(az))
                let d = radial * cos(elev) + up * sin(elev)
                let len = blade * (0.75 + 0.3 * rng.unit()) * (0.7 + 0.3 * sin(.pi * min(1, t * 1.3)))
                let topN = safeNormalize(simd_cross(side, d), fallback: up)
                let w = max(0.012, 0.032 * (blade / 0.6))
                let base = hub - up * (blade * 0.25 * t) + d * (trunkR * 0.5)
                // Old blades sag; live ones are stiff.
                let sag = len * p.droop * t * t * 0.6
                let mid = base + d * (len * 0.5) - up * (sag * 0.3)
                let end = base + d * len - up * sag
                let dying = max(0, t - 0.8) * 3
                let c0 = mixClamped(look.leaf, p.dead, dying) * (0.85 + 0.25 * rng.unit())
                let c1 = mixClamped(p.leafTip, p.dead, dying)
                // A shallow keel: the blade is a V in section.
                let keel = topN * (w * 0.35)
                palmQuad(&s, world, base - side * (w * 0.7), base + side * (w * 0.7), mid + side * w + keel, mid - side * w + keel,
                         hint: topN, c0, c0)
                palmTri(&s, world, mid - side * w + keel, mid + side * w + keel, end, hint: topN, c0, c0, c1)
            }
            s.cardMark = prevCard; s.foliageMark = prevMark; s.debugClass = prevDbg
        }
    }

    // MARK: - Primitives

    /// A triangle wound so its face looks toward `hint` (the leaf's top side), shaded with a normal
    /// leaning toward that side — a leaf group is drawn two-sided, so the winding only has to be
    /// consistent with the normal it carries.
    private static func palmTri(_ s: inout Soup, _ world: simd_float4x4,
                                _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, hint: SIMD3<Float>,
                                _ ca: SIMD3<Float>, _ cb: SIMD3<Float>, _ cc: SIMD3<Float>) {
        let g = simd_cross(b - a, c - a)
        guard simd_length_squared(g) > 1e-12 else { return }
        let flip = simd_dot(g, hint) < 0
        let gn = simd_normalize(flip ? -g : g)
        let n = xfDir(world, safeNormalize(gn * 0.7 + hint * 0.3, fallback: gn))
        let A = xf(world, a), B = xf(world, flip ? c : b), C = xf(world, flip ? b : c)
        s.tri(A, B, C, n0: n, n1: n, n2: n, c0: ca, c1: flip ? cc : cb, c2: flip ? cb : cc)
    }

    private static func palmQuad(_ s: inout Soup, _ world: simd_float4x4,
                                 _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ d: SIMD3<Float>,
                                 hint: SIMD3<Float>, _ c0: SIMD3<Float>, _ c1: SIMD3<Float>) {
        palmTri(&s, world, a, b, c, hint: hint, c0, c0, c1)
        palmTri(&s, world, a, c, d, hint: hint, c0, c1, c1)
    }

    /// A tapered tube through `path` (local space) with a colour per station — a ringed stem.
    private static func palmTube(_ s: inout Soup, world: simd_float4x4, path: [SIMD3<Float>], radii: [Float],
                                 colors: [SIMD3<Float>], sides: Int) {
        guard path.count >= 2, path.count == radii.count, colors.count == path.count else { return }
        var pos: [[SIMD3<Float>]] = [], nor: [[SIMD3<Float>]] = []
        for i in path.indices {
            let t = i == 0 ? path[1] - path[0]
                  : (i == path.count - 1 ? path[i] - path[i - 1] : path[i + 1] - path[i - 1])
            let tan = safeNormalize(t, fallback: SIMD3(0, 1, 0))
            let ref: SIMD3<Float> = abs(tan.y) < 0.95 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
            let u = simd_normalize(simd_cross(ref, tan))
            let v = simd_cross(tan, u)
            var rp: [SIMD3<Float>] = [], rn: [SIMD3<Float>] = []
            for q in 0 ..< sides {
                let a = 2 * Float.pi * Float(q) / Float(sides)
                let dir = u * cos(a) + v * sin(a)
                rp.append(xf(world, path[i] + dir * radii[i]))
                rn.append(xfDir(world, dir))
            }
            pos.append(rp); nor.append(rn)
        }
        for i in 0 ..< path.count - 1 {
            for q in 0 ..< sides {
                let q2 = (q + 1) % sides
                let a = pos[i][q], b = pos[i][q2], c = pos[i + 1][q2], d = pos[i + 1][q]
                // Wound outward — checked against the ring's own radial normal.
                let outward = simd_dot(simd_cross(b - a, c - a), nor[i][q] + nor[i][q2]) >= 0
                if outward {
                    s.tri(a, b, c, n0: nor[i][q], n1: nor[i][q2], n2: nor[i + 1][q2], c0: colors[i], c1: colors[i], c2: colors[i + 1])
                    s.tri(a, c, d, n0: nor[i][q], n1: nor[i + 1][q2], n2: nor[i + 1][q], c0: colors[i], c1: colors[i + 1], c2: colors[i + 1])
                } else {
                    s.tri(a, c, b, n0: nor[i][q], n1: nor[i + 1][q2], n2: nor[i][q2], c0: colors[i], c1: colors[i + 1], c2: colors[i])
                    s.tri(a, d, c, n0: nor[i][q], n1: nor[i + 1][q], n2: nor[i + 1][q2], c0: colors[i], c1: colors[i + 1], c2: colors[i + 1])
                }
            }
        }
    }
}
