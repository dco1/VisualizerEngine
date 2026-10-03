import Foundation
import simd

// A CONIFER, grown the way a conifer grows: a leader, WHORLS of primary branches, SPRIGS along every
// branch. This is the factory's second growth mode (the first, in `addTree`, is the broadleaf crown:
// arms off a trunk top + a volumetric leaf fill). A fir has neither a crown nor a fill — and the
// earlier attempt at a fir built outside this factory (a stack of closed "tier" skirts) read as
// layers of cut fabric however it was tuned, because a skirt is one big smooth surface and a fir has
// none.
//
// Everything below is a branch or a sprig at the scale a person reads it at:
//
// * **Whorls** every ~8 % of the canopy height, 8–9 branches low down thinning to 4–5 near the top,
//   each whorl phase-shifted by the golden angle so no two stack. Branch length follows the convex
//   cone ±12 %; a few are short or broken, so the silhouette is ragged rather than a scalloped circle.
// * **Branch curve** — leaves the trunk at a pitch (near-level low down, angled up near the top),
//   droops under its own weight, and the tip LIFTS: a fir branch's signature.
// * **Sprigs** — short lens-shaped branchlets, alternating along each branch, swept forward ~60°,
//   longest mid-branch, with a terminal sprig at the tip: the fishbone spray of a real fir branch.
//   Blades come from the SAME `LeafConstructor` + `TreeLeafStitch` every other tree leaf uses.
//
// Every dimension is a ratio of the canopy height, so a desk fir and a hall fir are one tree at two
// sizes. Anything that hangs on the tree (a string of lights, baubles, a garland) reads the SAME
// `ConiferSkeleton` — it attaches to branches that exist.
extension ForestTreeGeometry {

    // MARK: - Skeleton (pure data, local space: origin at the trunk base, +Y up)

    public struct ConiferBranch: Sendable {
        public var whorl: Int
        public var azimuth: Float
        public var path: [SIMD3<Float>]
        public var length: Float

        /// A point `t` (0 trunk … 1 tip) along the branch, linear between stations.
        public func point(at t: Float) -> SIMD3<Float> {
            let n = path.count - 1
            let x = max(0, min(1, t)) * Float(n)
            let i = min(n - 1, Int(x))
            let f = x - Float(i)
            return path[i] + (path[i + 1] - path[i]) * f
        }
        /// The unit direction of growth at `t`.
        public func tangent(at t: Float) -> SIMD3<Float> {
            let n = path.count - 1
            let i = min(n - 1, Int(max(0, min(1, t)) * Float(n)))
            let d = path[i + 1] - path[i]
            return simd_length(d) > 1e-9 ? simd_normalize(d) : SIMD3(0, 1, 0)
        }
        public var tip: SIMD3<Float> { path[path.count - 1] }
    }

    public struct ConiferSkeleton: Sendable {
        public var bottomY: Float
        public var apexY: Float
        public var baseRadius: Float
        public var convexity: Float
        public var look: ConiferLook
        public var branches: [ConiferBranch]
        /// Branch indices grouped by whorl, bottom to top, in azimuth order.
        public var whorls: [[Int]]

        public var height: Float { apexY - bottomY }
        public func y(atFraction f: Float) -> Float { bottomY + height * f }
        /// A slightly convex cone: a real fir holds its width a little further up than a straight one.
        public func radius(atFraction f: Float) -> Float { baseRadius * pow(max(0, 1 - f), convexity) }
    }

    /// Grow the skeleton. `density` is the Fullness slider (1 = default): more whorls and closer
    /// sprigs, never more triangles per sprig.
    public static func growConiferSkeleton(bottomY: Float, apexY: Float, baseRadius: Float,
                                           look c: ConiferLook, density: Float = 1,
                                           seed: UInt64) -> ConiferSkeleton {
        func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float { a + (b - a) * t }
        var rng = ForestRNG(seed: seed &+ 0xF12F_0001)
        var sk = ConiferSkeleton(bottomY: bottomY, apexY: apexY, baseRadius: baseRadius,
                                 convexity: c.convexity, look: c, branches: [], whorls: [])
        let H = sk.height
        // Size LOD: a small tree has no use for sprigs smaller than a pixel from any sane distance,
        // so below ~1 m of canopy the whorls and sprigs thin out (never below half).
        let d = max(0.5, min(2.2, density)) * max(0.5, min(1, (H / 1.0).squareRoot()))
        let spacing = c.whorlSpacing * H / pow(d, 0.55)
        let nW = max(5, Int((c.reach * H / spacing).rounded()))
        for w in 0 ..< nW {
            let f = (Float(w) + 0.35) / Float(nW) * c.reach
            let y0 = sk.y(atFraction: f)
            let count = max(6, Int((6.0 + 7.0 * (1 - f)).rounded()))
            let phase0 = Float(w) * 2.399963 + rng.unit() * 0.5
            var ring: [(az: Float, idx: Int)] = []
            for k in 0 ..< count {
                let az = phase0 + 2 * .pi * (Float(k) + (rng.unit() - 0.5) * 0.55) / Float(count)
                var L = sk.radius(atFraction: f) * (0.88 + 0.24 * rng.unit())
                if rng.unit() < 0.07 { L *= 0.6 }                     // a short, broken branch
                L = max(L, 0.06)
                let pitch = lerp(-0.10, 0.50, f) + (rng.unit() - 0.5) * 0.12
                let droop = lerp(0.34, 0.15, f) * (0.85 + 0.3 * rng.unit())
                let lift = lerp(0.52, 0.22, f)
                let ca = cos(az), sa = sin(az), tp = tan(pitch)
                // A little sideways sway: branches wander, they are not ruler-straight spokes.
                let swayAmp = L * (0.03 + 0.06 * rng.unit()) * (rng.unit() < 0.5 ? -1 : 1)
                let swayPhase = rng.unit() * 2 * .pi
                let path = (0 ... 7).map { i -> SIMD3<Float> in
                    let t = Float(i) / 7
                    let r = L * t
                    let y = y0 + L * (tp * t - droop * t * t + lift * pow(t, 4))
                    let sw = swayAmp * sin(t * 2.4 + swayPhase) * t
                    return SIMD3(ca * r - sa * sw, y, sa * r + ca * sw)
                }
                ring.append((az, sk.branches.count))
                sk.branches.append(ConiferBranch(whorl: w, azimuth: az, path: path, length: L))
            }
            sk.whorls.append(ring.sorted { fmodf($0.az, 2 * .pi) < fmodf($1.az, 2 * .pi) }.map(\.idx))
        }
        return sk
    }

    // MARK: - Emission

    /// A sprig: a short lens with a slightly pointed end — a needled branchlet seen as a card.
    public static let firSprigSilhouette = LeafSilhouette(name: "fir-sprig", hero: [
        (0.00, 0.50), (1.00, 0.00),
    ])

    /// Bake a conifer into the soup: trunk, branch tubes, sprigs. Positions are in the soup's world
    /// space (`world` places the trunk base).
    static func emitConifer(_ s: inout Soup, world: simd_float4x4, look: TreeLook,
                            conifer c: ConiferLook, site: TreeSite) {
        let H = site.height
        let bottomY = H * 0.12                       // a hand of bare trunk under the lowest whorl
        let sk = growConiferSkeleton(bottomY: bottomY, apexY: H, baseRadius: c.baseRadius * (H - bottomY),
                                     look: c, density: max(0.5, min(2.2, site.foliageScale / 0.30)),
                                     seed: site.seed)
        emitConiferWood(&s, world: world, sk: sk, look: look, trunkR: site.trunkR)
        emitConiferSprigs(&s, world: world, sk: sk, look: look, seed: site.seed)
    }

    /// The trunk and every primary branch, as tapered tubes through the skeleton's own paths.
    public static func emitConiferWood(_ s: inout Soup, world: simd_float4x4, sk: ConiferSkeleton,
                                       look: TreeLook, trunkR: Float, trunkBaseY: Float = 0) {
        let prev = s.debugClass; s.debugClass = 2
        defer { s.debugClass = prev }
        let H = sk.apexY
        let trunkPath = (0 ... 8).map { SIMD3<Float>(0, trunkBaseY + (H - trunkBaseY) * Float($0) / 8, 0) }
        let trunkRadii = (0 ... 8).map { i -> Float in
            let t = Float(i) / 8
            return trunkR * (t < 0.15 ? 1.20 - 0.2 * t / 0.15 : 1.0 - 0.88 * (t - 0.15) / 0.85)
        }
        emitConiferTube(&s, world: world, path: trunkPath, radii: trunkRadii, sides: 14, color: look.bark)
        var rng = ForestRNG(seed: UInt64(truncatingIfNeeded: Int(H * 1000)) &+ 0xB0A4D)
        for b in sk.branches {
            let r0 = sk.look.branchRadius * sk.height * max(0.45, min(1.2, b.length / max(0.1, 0.55 * sk.baseRadius)))
            let radii = (0 ..< b.path.count).map { r0 * (1 - 0.78 * Float($0) / Float(b.path.count - 1)) }
            emitConiferTube(&s, world: world, path: b.path, radii: radii, sides: 6,
                            color: look.bark * (0.85 + 0.30 * rng.unit()))
        }
    }

    /// A tapered tube through `path` (local space), welded rings with outward winding and smooth
    /// per-ring normals. Wood: emitted under the tree's wind context, so it sways with the trunk.
    static func emitConiferTube(_ s: inout Soup, world: simd_float4x4, path: [SIMD3<Float>],
                                radii: [Float], sides: Int, color: SIMD3<Float>) {
        guard path.count >= 2, path.count == radii.count else { return }
        var pos: [[SIMD3<Float>]] = [], nor: [[SIMD3<Float>]] = []
        for i in path.indices {
            let t = i == 0 ? path[1] - path[0]
                  : (i == path.count - 1 ? path[i] - path[i - 1] : path[i + 1] - path[i - 1])
            let tan = safeNormalize(t, fallback: SIMD3(0, 1, 0))
            let ref: SIMD3<Float> = abs(tan.y) < 0.95 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
            let u = simd_normalize(simd_cross(ref, tan))
            let v = simd_cross(tan, u)
            var rp: [SIMD3<Float>] = [], rn: [SIMD3<Float>] = []
            for k in 0 ..< sides {
                let a = 2 * Float.pi * Float(k) / Float(sides)
                let dir = u * cos(a) + v * sin(a)
                rp.append(xf(world, path[i] + dir * radii[i]))
                rn.append(simd_normalize((world * SIMD4<Float>(dir, 0)).xyz))
            }
            pos.append(rp); nor.append(rn)
        }
        for i in 0 ..< path.count - 1 {
            for k in 0 ..< sides {
                let k2 = (k + 1) % sides
                let a = pos[i][k], b = pos[i][k2], c = pos[i + 1][k2], d = pos[i + 1][k]
                let na = nor[i][k], nb = nor[i][k2], nc = nor[i + 1][k2], nd = nor[i + 1][k]
                s.tri(a, b, c, n0: na, n1: nb, n2: nc, c0: color, c1: color, c2: color)
                s.tri(a, c, d, n0: na, n1: nc, n2: nd, c0: color, c1: color, c2: color)
            }
        }
    }

    /// Sprigs along every branch, plus a terminal sprig at each tip. Triangles: 16 per sprig,
    /// single-sided (the leaf draw group is registered double-sided).
    public static func emitConiferSprigs(_ s: inout Soup, world: simd_float4x4, sk: ConiferSkeleton,
                                         look: TreeLook, seed: UInt64) {
        var rng = ForestRNG(seed: seed &+ 0xF12F_0002)
        let c = sk.look
        let H = sk.height
        let up = SIMD3<Float>(0, 1, 0)
        let prevDbg = s.debugClass; s.debugClass = 6
        let prevMark = s.foliageMark; s.foliageMark = 1
        defer { s.debugClass = prevDbg; s.foliageMark = prevMark }

        func sprig(base lb: SIMD3<Float>, dir ld: SIMD3<Float>, length: Float, ramp: Float, jitter: Float, roll: Float) {
            let base = xf(world, lb)
            let dir = simd_normalize((world * SIMD4<Float>(ld, 0)).xyz)
            let wUp = simd_normalize((world * SIMD4<Float>(up, 0)).xyz)
            var bn = wUp - dir * simd_dot(wUp, dir)
            bn = safeNormalize(bn, fallback: SIMD3(1, 0, 0))
            bn = simd_normalize(bn * cos(roll) + simd_cross(dir, bn) * sin(roll))
            let xax = simd_normalize(simd_cross(dir, bn))
            let prevCard = s.cardMark
            s.cardMark = s.nextCardID; s.nextCardID &+= 1
            defer { s.cardMark = prevCard }
            let k = ramp * jitter
            var stitch = TreeLeafStitch(bentNormal: bn,
                                        midribCol: c.needleShade * (0.9 * k),
                                        marginCol: mixv(c.needleShade, c.needleLit, 0.45) * k,
                                        tipCol: c.needleLit * (0.95 * k))
            swap(&stitch.soup, &s)
            LeafConstructor.emitBlade(
                into: &stitch,
                placement: LeafConstructor.Placement(position: base + dir * (length * 0.46),
                                                     xAxis: xax, yAxis: dir, bentNormal: bn,
                                                     width: length * c.sprigAspect, height: length),
                silhouette: firSprigSilhouette, tier: .hero, subdivisions: 1,
                fold: 0.22, curl: look.leafCurl, winding: .singleSided)
            swap(&stitch.soup, &s)
        }

        let sprigLen = c.sprigLength * H
        for b in sk.branches {
            let stations = max(4, Int((b.length * 0.94 / (c.sprigSpacing * H)).rounded()))
            for j in 0 ..< stations {
                let t = 0.04 + 0.94 * (Float(j) + 0.5) / Float(stations)
                let tan = b.tangent(at: t)
                let side = safeNormalize(simd_cross(up, tan), fallback: SIMD3(1, 0, 0))
                let lenFac = 0.70 + 0.30 * sin(.pi * (0.08 + 0.80 * t))
                let ramp = 0.55 + 0.75 * t                      // darker toward the trunk, lit toward the tip
                for sd: Float in [-1, 1] {
                    let tj = max(0.06, min(1.0, t + (rng.unit() - 0.5) * 0.5 / Float(stations)))
                    let a = 1.05 - 0.40 * t + (rng.unit() - 0.5) * 0.30
                    var d = tan * cos(a) + side * (sd * sin(a))
                    d.y += (rng.unit() - 0.35) * 0.32 - 0.10
                    sprig(base: b.point(at: tj), dir: simd_normalize(d),
                          length: sprigLen * lenFac * (0.80 + 0.40 * rng.unit()),
                          ramp: ramp, jitter: 0.82 + 0.36 * rng.unit(), roll: (rng.unit() - 0.5) * 1.0)
                }
                // A fourth, hanging below the branch: needles grow all round the shoot, and the
                // underside is what you see from a low angle.
                do {
                    let tj = max(0.06, min(1.0, t + (rng.unit() - 0.5) * 0.5 / Float(stations)))
                    var d = tan * 0.78 - up * 0.45 + side * ((rng.unit() - 0.5) * 0.9)
                    d = simd_normalize(d)
                    sprig(base: b.point(at: tj), dir: d, length: sprigLen * lenFac * (0.65 + 0.30 * rng.unit()),
                          ramp: ramp * 0.80, jitter: 0.80 + 0.30 * rng.unit(), roll: (rng.unit() - 0.5) * 1.2)
                }
                // A third sprig lifted out of the branch plane: a fir spray is a brush, not a comb —
                // seen end-on or from above it must still have needles.
                do {
                    let tj = max(0.06, min(1.0, t + (rng.unit() - 0.5) * 0.5 / Float(stations)))
                    var d = tan * 0.75 + up * 0.55 + side * ((rng.unit() - 0.5) * 0.7)
                    d = simd_normalize(d)
                    sprig(base: b.point(at: tj), dir: d, length: sprigLen * lenFac * (0.70 + 0.35 * rng.unit()),
                          ramp: ramp * 1.05, jitter: 0.82 + 0.36 * rng.unit(), roll: (rng.unit() - 0.5) * 1.2)
                }
            }
            sprig(base: b.point(at: 0.97), dir: b.tangent(at: 1.0), length: sprigLen * 1.25,
                  ramp: 1.25, jitter: 0.9 + 0.2 * rng.unit(), roll: 0)
        }

        // THE CORE: a dark cone of foliage well INSIDE the branches (half the envelope's radius), so
        // the gaps between sprigs show shadowed needle-mass — depth — not the room behind. A fir is a
        // solid mass seen through its own fringe; without this the tree read as a lattice of sprigs.
        // It is far enough in that it never reaches the silhouette.
        let segs = 14
        let shade = c.needleShade * 1.15
        let n0 = 9
        func ring(_ i: Int) -> (y: Float, r: Float) {
            let f = Float(i) / Float(n0)
            return (sk.y(atFraction: 0.01 + 0.97 * f), 0.52 * sk.radius(atFraction: 0.01 + 0.97 * f))
        }
        for i in 0 ..< n0 {
            let a = ring(i), b = ring(i + 1)
            for k in 0 ..< segs {
                let t0 = 2 * Float.pi * Float(k) / Float(segs), t1 = 2 * Float.pi * Float(k + 1) / Float(segs)
                func P(_ ra: (y: Float, r: Float), _ t: Float) -> SIMD3<Float> { xf(world, SIMD3(cos(t) * ra.r, ra.y, sin(t) * ra.r)) }
                // The cone's own slope: outward, tilted up by how fast the radius falls with height.
                let rise = max(1e-4, b.y - a.y)
                let ny = max(0, (a.r - b.r) / rise)
                func N(_ t: Float) -> SIMD3<Float> { simd_normalize((world * SIMD4<Float>(cos(t), ny, sin(t), 0)).xyz) }
                let p00 = P(a, t0), p01 = P(a, t1), p10 = P(b, t0), p11 = P(b, t1)
                // Wound OUTWARD: around the ring, then up is inward-facing, so the other way.
                s.tri(p00, p11, p01, n0: N(t0), n1: N(t1), n2: N(t1), c0: shade, c1: shade, c2: shade)
                s.tri(p00, p10, p11, n0: N(t0), n1: N(t0), n2: N(t1), c0: shade, c1: shade, c2: shade)
            }
        }
    }
}

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3(x, y, z) }
}
