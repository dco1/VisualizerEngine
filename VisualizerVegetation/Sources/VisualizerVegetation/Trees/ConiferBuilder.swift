import Foundation
import simd
import VisualizerMaterials

/// **The shared conifer builder** — one pine/spruce/fir generator for every consuming app.
///
/// It replaces two independent attempts at the same idea: Visualizer's UFO-scene `PineGeometry`
/// (SceneKit-only, noise-modulated cones) and the Daydream Christmas tree's tier stack. Both agree
/// on the one thing that matters about a conifer — its silhouette is a SOLID stack of drooping
/// bough TIERS, never a porcupine of radiating spikes — so that is what lives here, on `Mesh3`
/// with no SceneKit, and every species (spruce, generic, redwood, ponderosa, the Christmas fir)
/// is a `ConiferProfile` over the same tier stack.
///
/// **One envelope, many consumers.** The tier stack is built once per call; the canopy mesh, and
/// anything hung on the tree (baubles, a string of bulbs, snow, cones), read the tiers rather than
/// re-deriving a radius — move the envelope and they all move ([[feedback-single-source-of-truth]]).
///
/// Local space: X/Z plan about the trunk axis, Y up.
public enum ConiferBuilder {

    // MARK: - Species

    public enum Species: String, CaseIterable, Sendable {
        case spruce, generic, redwood, ponderosa
        /// A cut fir as sold for Christmas: dense, tiered, deep blue-green.
        case christmasTree
    }

    /// Everything that makes one species look unlike another. Pure data.
    public struct Profile: Sendable {
        // Trunk + where the canopy starts (fractions of total height).
        public var trunkHeightFrac: Double
        public var flareBase: Double, flareMid: Double, flareTop: Double   // × trunkRadius
        public var canopyBaseFrac: Double        // canopy bottom, as a fraction of the TRUNK height
        // Canopy envelope radius at its base: `trunkRadius × (core + spread)`.
        public var canopyCore: Double
        public var canopySpread: Double
        public var convexity: Double             // radius(h) = base × (1 − h)^convexity
        // Tier stack.
        public var tierCountSparse: Int
        public var tierCountLush: Int
        public var reach: Double                 // tier RIMS span this fraction of the canopy
        public var overlap: Double               // a tier's top reaches this many rim-spacings up
        public var lobeSpacing: Double           // metres of rim per bough tip (at 1.5 m tall)
        public var droop: Double                 // bough-tip hang, as a fraction of rim radius
        public var rimJitter: Double             // seeded per-tier radius variation
        // Colour (linear-ish albedo).
        public var needle: Vec3
        public var needleLift: Vec3              // brighter by this at the top tier
        public var bark: Vec3

        public static func of(_ s: Species) -> Profile {
            switch s {
            case .spruce:
                return Profile(trunkHeightFrac: 0.22, flareBase: 1.6, flareMid: 1.0, flareTop: 0.55,
                               canopyBaseFrac: 0.40, canopyCore: 0.8, canopySpread: 3.8, convexity: 1.0,
                               tierCountSparse: 8, tierCountLush: 12, reach: 0.90, overlap: 1.7,
                               lobeSpacing: 0.055, droop: 0.12, rimJitter: 0.10,
                               needle: Vec3(0.05, 0.17, 0.14), needleLift: Vec3(0.05, 0.06, 0.04),
                               bark: Vec3(0.14, 0.11, 0.09))
            case .generic:
                return Profile(trunkHeightFrac: 0.28, flareBase: 1.4, flareMid: 1.0, flareTop: 0.7,
                               canopyBaseFrac: 0.45, canopyCore: 1.4, canopySpread: 6.5, convexity: 1.1,
                               tierCountSparse: 6, tierCountLush: 9, reach: 0.88, overlap: 1.8,
                               lobeSpacing: 0.07, droop: 0.12, rimJitter: 0.14,
                               needle: Vec3(0.06, 0.15, 0.08), needleLift: Vec3(0.05, 0.06, 0.04),
                               bark: Vec3(0.16, 0.10, 0.07))
            case .redwood:
                return Profile(trunkHeightFrac: 0.55, flareBase: 2.0, flareMid: 1.1, flareTop: 0.5,
                               canopyBaseFrac: 0.55, canopyCore: 1.0, canopySpread: 8.5, convexity: 1.2,
                               tierCountSparse: 5, tierCountLush: 7, reach: 0.86, overlap: 1.7,
                               lobeSpacing: 0.09, droop: 0.14, rimJitter: 0.16,
                               needle: Vec3(0.07, 0.20, 0.10), needleLift: Vec3(0.05, 0.06, 0.04),
                               bark: Vec3(0.34, 0.17, 0.10))
            case .ponderosa:
                return Profile(trunkHeightFrac: 0.50, flareBase: 1.5, flareMid: 0.9, flareTop: 0.5,
                               canopyBaseFrac: 0.65, canopyCore: 0.9, canopySpread: 5.5, convexity: 1.3,
                               tierCountSparse: 4, tierCountLush: 6, reach: 0.84, overlap: 1.6,
                               lobeSpacing: 0.10, droop: 0.15, rimJitter: 0.22,
                               needle: Vec3(0.10, 0.22, 0.09), needleLift: Vec3(0.06, 0.07, 0.04),
                               bark: Vec3(0.28, 0.16, 0.08))
            case .christmasTree:
                return Profile(trunkHeightFrac: 0.85, flareBase: 1.0, flareMid: 0.7, flareTop: 0.45,
                               canopyBaseFrac: 0.10, canopyCore: 0, canopySpread: 13, convexity: 0.95,
                               tierCountSparse: 5, tierCountLush: 9, reach: 0.86, overlap: 1.6,
                               lobeSpacing: 0.055, droop: 0.10, rimJitter: 0.12,
                               needle: Vec3(0.04, 0.16, 0.10), needleLift: Vec3(0.03, 0.05, 0.03),
                               bark: Vec3(0.20, 0.13, 0.08))
            }
        }

        /// Tier count for a fullness 0…1+ (`foliageDensity` / `canopyFullness`).
        public func tierCount(fullness: Double) -> Int {
            let f = max(0, min(1, (fullness - 0.4) / 1.2 + 0.0))
            let lerped = Double(tierCountSparse) + Double(tierCountLush - tierCountSparse) * f
            return max(tierCountSparse, min(tierCountLush, Int(lerped.rounded())))
        }
    }

    // MARK: - Envelope

    /// The canopy's overall cone — where it starts, where the tip is, how wide the base tier is.
    public struct Envelope: Sendable {
        public var bottomY: Double
        public var apexY: Double
        public var baseRadius: Double
        public var convexity: Double

        public init(bottomY: Double, apexY: Double, baseRadius: Double, convexity: Double = 0.95) {
            self.bottomY = bottomY; self.apexY = apexY
            self.baseRadius = baseRadius; self.convexity = convexity
        }
        public var height: Double { apexY - bottomY }
        public func y(atFraction h: Double) -> Double { bottomY + height * h }
        /// Slightly convex cone: a real fir holds its width a little further up than a straight cone.
        public func radius(atFraction h: Double) -> Double { baseRadius * pow(max(0, 1 - h), convexity) }
    }

    // MARK: - Tiers

    /// One bough tier: a closed skirt. Its TOP surface falls from the trunk to a scalloped rim with
    /// a gravity droop that grows toward the tips; its UNDERSIDE climbs back in to the trunk; a short
    /// inner wall closes it. Watertight, so the canopy is a set of solids, not a cardboard shell.
    public struct Tier: Sendable {
        public var yTop: Double          // where the top surface meets the trunk
        public var yRim: Double          // rim height before the tips droop
        public var yUnder: Double        // where the underside meets the trunk
        public var rInner: Double        // top surface's inner radius (0 = the tree's tip)
        public var rUnderInner: Double   // underside's inner radius
        public var rRim: Double          // rim radius at a bough TIP
        public var lobes: Int            // bough tips around the rim
        public var phase: Double
        public var noiseA: Double
        public var noiseB: Double
        public var droop: Double         // how far a bough tip hangs below the rim

        /// 1 at a bough tip, 0 in the notch between two — sharp tips, broad notches.
        public func lobe(_ theta: Double) -> Double {
            pow(0.5 + 0.5 * cos(Double(lobes) * (theta - phase)), 2.5)
        }
        /// Seeded low-frequency wobble so no two tiers (and no two trees) share a rim.
        public func noise(_ theta: Double) -> Double {
            0.6 * sin(3 * theta + noiseA) + 0.4 * sin(7 * theta + noiseB)
        }
        public func rimRadius(_ theta: Double) -> Double {
            rRim * (0.78 + 0.22 * lobe(theta)) * (1 + 0.05 * noise(theta))
        }
        public func rimY(_ theta: Double) -> Double { yRim - droop * lobe(theta) }

        /// A point on the top surface, `u` 0 at the trunk → 1 at the rim. `pow(u, 1.3)` keeps the
        /// bough shallow near the trunk and steep at the tip — it droops, it does not tent.
        public func top(_ u: Double, _ theta: Double) -> Vec3 {
            let r = rInner + (rimRadius(theta) - rInner) * u
            let y = yTop - (yTop - yRim) * pow(u, 1.3) - droop * lobe(theta) * u * u
            return Vec3(cos(theta) * r, y, sin(theta) * r)
        }
        /// A point on the underside, `v` 0 at the rim → 1 at the trunk.
        public func under(_ v: Double, _ theta: Double) -> Vec3 {
            let r = rimRadius(theta) + (rUnderInner - rimRadius(theta)) * v
            let y = rimY(theta) + (yUnder - rimY(theta)) * pow(v, 0.7)
            return Vec3(cos(theta) * r, y, sin(theta) * r)
        }
    }

    /// The tier stack for an envelope. `lobeSpacing` is metres of rim per bough tip.
    public static func tiers(envelope env: Envelope, profile: Profile, count n: Int,
                             lobeSpacing: Double? = nil, rng: inout VegetationRNG) -> [Tier] {
        let spacing = profile.reach / Double(n)
        let lobeStep = lobeSpacing ?? profile.lobeSpacing
        return (0 ..< n).map { i in
            let hb = Double(i) * spacing
            let isTip = i == n - 1
            // Each tier's top reaches `overlap` rim-spacings up, well past the next tier's rim — the
            // overlap is what makes the stack read as layered boughs rather than a pagoda of discs.
            let ht = isTip ? 1.0 : min(1.0, hb + profile.overlap * spacing)
            let rRim = max(0.04, env.radius(atFraction: hb)) * ((1 - profile.rimJitter / 2) + profile.rimJitter * rng.unit())
            let yRim = env.y(atFraction: hb)
            let yTop = env.y(atFraction: ht)
            let inner = max(0.02, rRim * 0.10)
            return Tier(yTop: yTop, yRim: yRim, yUnder: yRim + (yTop - yRim) * 0.30,
                        rInner: isTip ? 0 : inner, rUnderInner: inner,
                        rRim: rRim, lobes: max(5, Int((rRim / lobeStep).rounded())),
                        phase: rng.unit() * 2 * .pi,
                        noiseA: rng.unit() * 2 * .pi, noiseB: rng.unit() * 2 * .pi,
                        droop: rRim * profile.droop)
        }
    }

    // MARK: - Canopy

    /// Azimuth segments per tier — enough for ~5 per bough tip on the widest tier.
    public static let segments = 64

    /// The needle canopy: every tier as a closed, smoothed skirt. Smoothed PER TIER (tiers overlap,
    /// and averaging normals across two tiers' coincident points would blend unrelated surfaces);
    /// the 40° crease keeps each rim a crisp edge.
    ///
    /// `creaseDegrees` 62: at the old 40° the sharp peak of each bough tip stayed a hard fold — a
    /// visible vertical crease down every lobe (rubric 8/9 read it as folded card). Above ~60° a tip
    /// shades round, while the rim-to-underside seam (≈90°) is still a crisp edge.
    public static let creaseDegrees: Double = 62

    public static func canopy(_ tiers: [Tier]) -> Mesh3 {
        let s = segments
        let thetas = (0 ..< s).map { Double($0) / Double(s) * 2 * .pi }
        var canopy = Mesh3()
        for t in tiers {
            // Rings, inside → out → back in: top surface u = 0…1, then the underside v = ⅓…1
            // (v = 0 IS the rim, already the top surface's last ring — shared, so the seam welds).
            let topRings = [0.0, 0.42, 0.76, 1.0].map { u in thetas.map { t.top(u, $0) } }
            let underRings = [0.45, 1.0].map { v in thetas.map { t.under(v, $0) } }
            var m = Mesh3()
            func strip(_ a: [Vec3], _ b: [Vec3], outward: (Double) -> Vec3) {
                for j in 0 ..< s {
                    let k = (j + 1) % s
                    let mid = (thetas[j] + (k == 0 ? 2 * .pi : thetas[k])) / 2
                    m.addQuad(a[j], a[k], b[k], b[j], outward: outward(mid))
                }
            }
            // Top surface faces up-and-out; the underside down-and-in; the inner wall toward the axis.
            for r in 0 ..< topRings.count - 1 {
                strip(topRings[r], topRings[r + 1]) { Vec3(cos($0), 1, sin($0)) }
            }
            strip(topRings[topRings.count - 1], underRings[0]) { Vec3(-0.3 * cos($0), -1, -0.3 * sin($0)) }
            strip(underRings[0], underRings[1]) { Vec3(-0.3 * cos($0), -1, -0.3 * sin($0)) }
            strip(underRings[1], topRings[0]) { Vec3(-cos($0), -0.2, -sin($0)) }
            canopy.append(m.smoothed(creaseDegrees: creaseDegrees))
        }
        return canopy
    }

    // MARK: - Painted needles (per-vertex albedo)

    /// Needle albedo ramp: a dark blue-green core in the bough's shade, lit tips at its edge.
    /// `Profile.needle` is the species' mid colour; the ramp spans ×0.45 → ×1.35 of it, with a
    /// cool shift in the shade (rubric 7: spruce/fir read BLUE-green, never leafy yellow-green).
    public static func needleRamp(_ p: Profile, _ t: Double) -> Vec3 {
        let shade = Vec3(p.needle.x * 0.60, p.needle.y * 0.78, p.needle.z * 1.15)
        let lit = Vec3(p.needle.x * 1.45 + p.needleLift.x * 0.5, p.needle.y * 1.40 + p.needleLift.y * 0.5,
                       p.needle.z * 1.10)
        let k = max(0, min(1, t))
        return Vec3(shade.x + (lit.x - shade.x) * k, shade.y + (lit.y - shade.y) * k, shade.z + (lit.z - shade.z) * k)
    }

    /// Albedo for a point on one tier, as a pure function of where it sits on that tier: shaded at the
    /// trunk, brightening toward the tip (new growth), a lighter ridge down each bough, faint
    /// striations along it, and the underside in deep shade.
    public static func tint(_ t: Tier, _ p: Vec3, profile: Profile) -> Vec3 {
        let th = atan2(p.z, p.x)
        let r = (p.x * p.x + p.z * p.z).squareRoot()
        let rim = t.rimRadius(th)
        let u = max(0, min(1, (r - t.rInner) / max(1e-6, rim - t.rInner)))
        let depth = max(0, t.top(u, th).y - p.y)                  // below the top surface ⇒ underside
        let under = min(1, depth / max(1e-6, 0.06 * t.rRim))
        let ridge = t.lobe(th)                                     // 1 down a bough's spine, 0 in the notch
        let stria = 0.5 + 0.5 * sin(Double(t.lobes) * 7 * (th - t.phase))
        var k = pow(u, 1.1) * (0.55 + 0.45 * ridge) + 0.08 * stria * u
        k = max(0, min(1, k))
        let c = needleRamp(profile, k)
        let f = 1 - 0.62 * under
        return Vec3(c.x * f, c.y * f, c.z * f)
    }

    /// The tiers AND the per-vertex albedo for them, index-aligned with the returned mesh (smoothed
    /// per tier exactly as `canopy(_:)` does — same geometry, plus a colour for every vertex).
    public static func paintedCanopy(_ tiers: [Tier], profile: Profile) -> (mesh: Mesh3, colors: [Vec3]) {
        var mesh = Mesh3()
        var colors: [Vec3] = []
        for t in tiers {
            let m = canopy([t])
            colors.append(contentsOf: m.positions.map { tint(t, $0, profile: profile) })
            mesh.append(m)
        }
        return (mesh, colors)
    }

    /// Albedo for the needle fringe: the lit end of the ramp, a little brighter higher up the tree.
    public static func fringeColors(_ fringe: Mesh3, profile: Profile, apexY: Double, bottomY: Double) -> [Vec3] {
        fringe.positions.map { p in
            let h = max(0, min(1, (p.y - bottomY) / max(1e-6, apexY - bottomY)))
            return needleRamp(profile, 0.72 + 0.28 * h)
        }
    }

    // MARK: - Needle fringe

    /// **The ragged edge.** A tier's rim is a smooth scallop — a lampshade — until needles break it:
    /// a conifer's silhouette is fuzz, not a line. This lays the shared `NeedleFascicle` (the same
    /// arrangement every needled plant uses, so it inherits `LeafConstructor`'s fold, curl and
    /// winding contract) along every tier's rim, pointing outward and down the way a bough tip's
    /// needles hang, plus a sparser second row a little way up the bough that overlaps the rim row
    /// so the join between surface and fringe never shows.
    ///
    /// Budget is the lever, not the look: `needles` per fascicle × 24 triangles (double-sided
    /// shell) × one fascicle per `spacing` metres of rim. Defaults land ~7 k triangles on a 1.8 m
    /// tree. Deterministic per seed — it draws from the caller's `rng`, after the tiers.
    ///
    /// Cards are OPEN surfaces, so a mesh carrying a fringe is not watertight: audit `canopy(_:)`
    /// (the closed skirts) for that, and the fringe for winding.
    public static func fringe(_ tiers: [Tier], spacing: Double = 0.07, needles: Int = 2,
                              length: Double = 0.07, rng: inout VegetationRNG) -> Mesh3 {
        var m = Mesh3()
        let up = Vec3(0, 1, 0)
        func emit(_ at: Vec3, _ dir: Vec3, _ len: Double, splay: Double) {
            NeedleFascicle.emit(into: &m, sheath: at, axis: normalize3(dir), reference: up, count: needles,
                                length: len, aspect: 0.055, splayRadians: splay, curl: 0.08,
                                winding: .doubleSidedShell)
        }
        for t in tiers {
            // Rim row: one fascicle every `spacing` metres of rim, bunched at the bough tips.
            let n = max(8, Int((2 * .pi * t.rRim / spacing).rounded()))
            for j in 0 ..< n {
                let th = Double(j) / Double(n) * 2 * .pi + (rng.unit() - 0.5) * 0.04
                let lobe = t.lobe(th)
                let at = t.top(1.0, th)
                // Out along the radius, falling harder where the bough tip droops.
                let dir = Vec3(cos(th), -(0.30 + 0.55 * lobe), sin(th))
                emit(at, dir, length * (0.8 + 0.5 * lobe) * (0.85 + 0.3 * rng.unit()), splay: 0.34)
            }
            // Second row, up the bough: shorter, riding the surface outward and a touch up.
            let n2 = max(6, Int((2 * .pi * t.rRim * 0.82 / (spacing * 2)).rounded()))
            for j in 0 ..< n2 {
                let th = (Double(j) + rng.unit()) / Double(n2) * 2 * .pi
                let at = t.top(0.82, th)
                let dir = Vec3(cos(th), 0.05 - 0.25 * t.lobe(th), sin(th))
                emit(at, dir, length * 0.7 * (0.85 + 0.3 * rng.unit()), splay: 0.30)
            }
        }
        return m
    }

    // MARK: - A whole standalone tree

    /// Everything a caller can tune on one tree. The base of the trunk sits on y = 0.
    public struct Spec: Sendable {
        public var height: Double = 6.0
        public var trunkRadius: Double = 0.18
        public var seed: UInt64 = 1
        public var species: Species = .generic
        /// Tier count; 0 uses the species default for `canopyFullness`.
        public var canopyLayers: Int = 0
        /// 0…1.2. Scales the base radius (thin → full).
        public var canopyFullness: Double = 1.0
        /// 0…1. Clumps of snow along the upper surface of the tiers.
        public var snowAmount: Double = 0
        /// 0…1. Drops the lowest tiers and thins the rest, and grows dead twigs.
        public var ageWear: Double = 0
        public var pineconeCount: Int = 0
        public init(height: Double = 6, trunkRadius: Double = 0.18, seed: UInt64 = 1,
                    species: Species = .generic, canopyLayers: Int = 0, canopyFullness: Double = 1,
                    snowAmount: Double = 0, ageWear: Double = 0, pineconeCount: Int = 0) {
            self.height = height; self.trunkRadius = trunkRadius; self.seed = seed
            self.species = species; self.canopyLayers = canopyLayers
            self.canopyFullness = canopyFullness; self.snowAmount = snowAmount
            self.ageWear = ageWear; self.pineconeCount = pineconeCount
        }
    }

    public struct Tree: Sendable {
        public var trunk: Mesh3
        public var canopy: Mesh3
        public var snow: Mesh3
        public var cones: Mesh3
        public var deadTwigs: Mesh3
        public var tiers: [Tier]
        public var envelope: Envelope
        public var profile: Profile
        public var needleColor: Vec3
        public var barkColor: Vec3
    }

    public static func envelope(for spec: Spec) -> Envelope {
        let p = Profile.of(spec.species)
        let trunkH = spec.height * p.trunkHeightFrac
        let base = spec.trunkRadius * (p.canopyCore + p.canopySpread) * max(0.4, min(1.2, spec.canopyFullness))
        return Envelope(bottomY: trunkH * p.canopyBaseFrac, apexY: spec.height,
                        baseRadius: base, convexity: p.convexity)
    }

    /// Build one standalone tree (trunk on y = 0). Deterministic per `spec`.
    public static func build(_ spec: Spec) -> Tree {
        let profile = Profile.of(spec.species)
        var rng = VegetationRNG(spec.seed)
        let env = envelope(for: spec)
        let n = spec.canopyLayers > 0 ? spec.canopyLayers : profile.tierCount(fullness: spec.canopyFullness)
        var tiers = tiers(envelope: env, profile: profile, count: max(3, n),
                          lobeSpacing: profile.lobeSpacing * max(0.5, spec.height / 1.5), rng: &rng)
        // An old tree loses its lowest limbs first.
        if spec.ageWear > 0.55 {
            let drop = Int(Double(tiers.count) * spec.ageWear * 0.25)
            if drop > 0, tiers.count - drop >= 3 { tiers.removeFirst(drop) }
        }
        var snow = Mesh3(), cones = Mesh3(), twigs = Mesh3()
        if spec.snowAmount > 0.001 { snow = snowClumps(tiers, amount: spec.snowAmount, rng: &rng) }
        if spec.pineconeCount > 0 { cones = pinecones(tiers, count: spec.pineconeCount, trunkRadius: spec.trunkRadius, rng: &rng) }
        if spec.ageWear > 0.01 { twigs = deadTwigsMesh(env, spec: spec, rng: &rng) }
        return Tree(trunk: trunk(spec, profile: profile, topY: max(env.y(atFraction: 0.9), spec.height * 0.9)),
                    canopy: canopy(tiers), snow: snow, cones: cones, deadTwigs: twigs,
                    tiers: tiers, envelope: env, profile: profile,
                    needleColor: profile.needle, barkColor: profile.bark)
    }

    /// Root flare → main trunk → upper trunk, as one revolved taper.
    public static func trunk(_ spec: Spec, profile p: Profile, topY: Double) -> Mesh3 {
        let r = spec.trunkRadius
        let flareH = max(0.02, topY * 0.04)
        var m = Mesh3()
        m.revolve(profile: [
            .init(r: r * p.flareBase, y: 0),
            .init(r: r * p.flareMid * 1.05, y: flareH),
            .init(r: r * p.flareMid * 0.8, y: topY * 0.35),
            .init(r: r * p.flareTop, y: topY),
        ], segments: 24, capBottom: true, capTop: true)
        return m.smoothed(creaseDegrees: 50)
    }

    /// Squashed snow piles draped on the outer top surface of the tiers (upper tiers heavier).
    public static func snowClumps(_ tiers: [Tier], amount: Double, rng: inout VegetationRNG) -> Mesh3 {
        var m = Mesh3()
        let n = Double(max(1, tiers.count - 1))
        for (i, t) in tiers.enumerated() {
            let mix = min(1, amount) * (0.30 + 0.70 * Double(i) / n)
            guard mix > 0.08 else { continue }
            let clumps = max(5, Int(mix * 18 * t.rRim / 0.55))
            for _ in 0 ..< clumps {
                let theta = rng.unit() * 2 * .pi
                let p = t.top(0.55 + 0.4 * rng.unit(), theta)
                let rad = max(0.02, t.rRim * (0.10 + 0.08 * rng.unit()) * mix)
                var b = Mesh3()
                b.revolve(profile: (0 ... 4).map { k -> Mesh3.ProfilePoint in
                    let a = (Double(k) / 4 - 0.5) * .pi
                    return .init(r: k == 0 || k == 4 ? 0 : cos(a) * rad * 1.4, y: sin(a) * rad * 0.45)
                }, segments: 10, capBottom: false, capTop: false)
                m.append(b.smoothed(creaseDegrees: 90).translated(by: Vec3(p.x, p.y + rad * 0.25, p.z)))
            }
        }
        return m
    }

    /// Teardrop cones hanging under the lower tiers' bough tips.
    public static func pinecones(_ tiers: [Tier], count: Int, trunkRadius: Double, rng: inout VegetationRNG) -> Mesh3 {
        var m = Mesh3()
        let lower = Array(tiers.prefix(max(1, tiers.count / 2)))
        for _ in 0 ..< count {
            let t = lower[Int(rng.unit() * Double(lower.count)) % lower.count]
            let theta = t.phase + 2 * .pi * Double(Int(rng.unit() * Double(t.lobes))) / Double(t.lobes)
            let r = t.rimRadius(theta) * 0.8
            let len = trunkRadius * 1.3, rad = trunkRadius * 0.30
            var c = Mesh3()
            c.revolve(profile: (0 ... 6).map { k -> Mesh3.ProfilePoint in
                let f = Double(k) / 6                       // 0 stem end → 1 blunt tip
                let w = k == 0 || k == 6 ? 0 : rad * sin(pow(f, 0.7) * .pi) * (1.1 - 0.3 * f)
                return .init(r: w, y: -len * f)
            }, segments: 10, capBottom: false, capTop: false)
            m.append(c.smoothed(creaseDegrees: 90).translated(by: Vec3(cos(theta) * r, t.rimY(theta) - rad, sin(theta) * r)))
        }
        return m
    }

    /// Bare grey stubs poking out through the canopy of an old tree.
    public static func deadTwigsMesh(_ env: Envelope, spec: Spec, rng: inout VegetationRNG) -> Mesh3 {
        var m = Mesh3()
        for _ in 0 ..< Int(spec.ageWear * 5) {
            let h = spec.height * (0.04 + 0.05 * rng.unit())
            let r0 = spec.trunkRadius * (0.4 + 0.3 * rng.unit())
            let a = rng.unit() * 2 * .pi
            let y = env.y(atFraction: 0.3 + 0.55 * rng.unit())
            let rad = max(spec.trunkRadius * 1.5, env.radius(atFraction: 0.4) * 0.5)
            var t = Mesh3()
            t.revolve(profile: [.init(r: r0, y: 0), .init(r: spec.trunkRadius * 0.06, y: h)],
                      segments: 8, capBottom: true, capTop: true)
            // Lean the stub outward ~40° from vertical, toward its azimuth.
            let out = Vec3(cos(a), 0, sin(a))
            t = t.rotatedZ(-0.7).rotatedY(-a)
            m.append(t.translated(by: Vec3(out.x * rad, y, out.z * rad)))
        }
        return m
    }
}
