import Foundation
import simd
import VisualizerMaterials

/// A HERO houseplant leaf built as a SHEET — a structured grid over the whole blade — rather than as
/// `LeafConstructor`'s spine-to-margin STRIP.
///
/// **Why a second blade constructor exists.** The strip has exactly two vertices across each half of
/// a blade: one on the midrib and one on the margin. That is the right economy for a tree canopy of
/// fifty thousand leaves, and it is the reason a houseplant leaf seen from a metre away read as a
/// paper cutout:
///  - **No midrib.** With nothing between the spine and the margin, a pale rib can only be a
///    gradient across the whole half-blade — the old fix was a separate ribbon mesh floating proud
///    of the leaf, which read as "a painted graphic stripe".
///  - **A V, not a leaf.** The cup (`fold`) lifts the margin in proportion to `u`, so every cross
///    section is two straight lines meeting at the midrib — a folded card, never a curved blade.
///  - **No splits, no holes.** A half-margin `u(v)` is one number per station; it cannot say "this
///    blade is cut through to within a hand of the midrib here", so a monstera could only ever be a
///    lettuce with a wavy edge (DH-0775's two failed silhouette rounds).
///
/// The sheet is a grid in the blade's own flat coordinates — `x` across and `y` along, in units of
/// the blade's length — lifted onto a curved `Surface` (cup, droop, ripple, midrib channel, the
/// pucker between lateral veins, twist). Interior columns give the rib a crisp band and the blade a
/// round section; a monstera's splits and holes are just cells of the grid left out
/// (`monsteraCells`). Every vertex is PAINTED as it is emitted (`PaintedMesh`), because only here is
/// it known which leaf, which face, and how near a vein each vertex is.
///
/// Both faces are emitted as geometry (the potted-plant convention — the meshes are registered
/// single-sided) so the underside can wear its own, paler paint.
public enum LeafSheet {

    /// One point of the flat blade: where it is, and how near it lies to a lateral vein.
    public struct Site: Sendable, Equatable {
        /// Across the blade, in blade lengths, signed (+ one half, − the other). 0 on the midrib.
        public var x: Double
        /// Along the blade, in blade lengths: 0 where the petiole meets it, 1 at the tip. A
        /// heart-shaped base's lobes reach a little below 0.
        public var y: Double
        /// 0 … 1 — how close this point sits to a lateral vein's line (1 on the vein).
        public var vein: Double
        public init(x: Double, y: Double, vein: Double = 0) { self.x = x; self.y = y; self.vein = vein }
    }

    /// The curved surface the flat blade is lifted onto. Lengths in BLADE units (× `length` = metres).
    public struct Surface: Sendable {
        /// Where the petiole meets the blade — the blade's (0, 0).
        public var base: Vec3
        /// Across the blade (the direction of +x).
        public var xAxis: Vec3
        /// Along the midrib, base → tip.
        public var yAxis: Vec3
        /// The UPPER face — the one the leaf presents to the light.
        public var normal: Vec3
        /// Metres per blade unit: the blade's length.
        public var length: Double
        /// The blade's widest half-width, in blade units — normalises `x` for the cup and ripple.
        public var halfWidth: Double
        /// Margin lift at the widest point (+ = margins up, a cupped leaf; − = a domed one).
        public var cup: Double = 0
        /// The cup's cross-section: 1 is the strip's straight V, 2 a round U.
        public var cupPower: Double = 1.7
        /// How far the tip hangs below the flat blade (a self-weighted cantilever, ∝ y²).
        public var droop: Double = 0
        /// Depth of the channel the midrib runs in on the upper face.
        public var midribGroove: Double = 0
        /// Half-width of that channel.
        public var midribHalfWidth: Double = 0.02
        /// Amplitude of the margin's undulation — zero on the midrib, full at the margin.
        public var ripple: Double = 0
        /// Undulations along the blade.
        public var rippleWaves: Double = 3
        public var ripplePhase: Double = 0
        /// How far the blade bulges up between its lateral veins (a leathery, quilted leaf).
        public var pucker: Double = 0
        /// Total twist about the midrib, base → tip (radians).
        public var twist: Double = 0

        public init(base: Vec3, xAxis: Vec3, yAxis: Vec3, normal: Vec3, length: Double, halfWidth: Double) {
            self.base = base
            self.xAxis = normalize3(xAxis)
            self.yAxis = normalize3(yAxis)
            self.normal = normalize3(normal)
            self.length = length
            self.halfWidth = max(1e-6, halfWidth)
        }

        /// The lifted point for a site of the flat blade.
        public func point(_ site: Site) -> Vec3 {
            let x = site.x, y = site.y
            let s = x / halfWidth
            let a = min(abs(s), 1.4)
            let v = max(0, y)
            var lift = cup * pow(a, cupPower)
            lift += ripple * a * a * sin(2 * Double.pi * rippleWaves * v + ripplePhase)
            if midribGroove != 0 {
                let g = max(0, 1 - abs(x) / max(1e-6, midribHalfWidth))
                lift -= midribGroove * g * g * (3 - 2 * g) * min(1, v * 6) * (1 - 0.6 * v)
            }
            if pucker != 0 {
                // Tissue between two veins domes up; on a vein, on the midrib and at the margin it
                // is held down — so the bulge is zero at all three.
                lift += pucker * (1 - site.vein) * 4 * a * max(0, 1 - a) * min(1, v * 5)
            }
            let drop = droop * v * v
            let ang = twist * v
            let xa = xAxis * cos(ang) + normal * sin(ang)
            let na = normal * cos(ang) - xAxis * sin(ang)
            return base + (yAxis * y + xa * x + na * lift) * length - normal * (drop * length)
        }

        /// The upper face's direction at a site (finite differences of `point`), turned to agree
        /// with `normal` — so the answer does not hang on which way round the caller's axes are.
        public func upper(at site: Site) -> Vec3 {
            let e = 0.004
            let py = point(Site(x: site.x, y: site.y + e, vein: site.vein))
                   - point(Site(x: site.x, y: site.y - e, vein: site.vein))
            let px = point(Site(x: site.x + e, y: site.y, vein: site.vein))
                   - point(Site(x: site.x - e, y: site.y, vein: site.vein))
            let n = cross3(px, py)
            guard len3(n) > 1e-14 else { return normal }
            let u = normalize3(n)
            return dot3(u, normal) >= 0 ? u : Vec3(-u.x, -u.y, -u.z)
        }
    }

    /// Paint for one vertex: its site and its face.
    public typealias Paint = (Site, LeafFace) -> Vec3

    /// Emit one cell of the flat blade — corners in order round the cell — on BOTH faces.
    public static func emitCell(into p: inout PaintedMesh, _ q: [Site], surface: Surface, paint: Paint) {
        guard q.count == 4 else { return }
        let P = q.map { surface.point($0) }
        let centre = Site(x: (q[0].x + q[2].x) / 2, y: (q[0].y + q[2].y) / 2,
                          vein: (q[0].vein + q[2].vein) / 2)
        let up = surface.upper(at: centre)
        for face in [LeafFace.upper, .lower] {
            let out = face == .upper ? up : Vec3(-up.x, -up.y, -up.z)
            let c = q.map { paint($0, face) }
            p.addQuad(P[0], P[1], P[2], P[3], colors: c[0], c[1], c[2], c[3], outward: out)
        }
    }

    // MARK: - Lateral veins

    /// A pinnate set of lateral veins: `pairs` veins per side leaving the midrib between `from` and
    /// `to` (fractions of the blade length), each running out at `angle` from the midrib and bending
    /// toward the tip by `sweep` as it nears the margin — the fiddle-leaf fig's pale herringbone.
    public struct LateralVeins: Sendable, Equatable {
        public var pairs: Int
        public var from: Double
        public var to: Double
        /// Angle between a vein and the midrib where it leaves it (radians).
        public var angle: Double
        /// Upward bend of a vein's outer end, blade units per (blade unit)² of reach.
        public var sweep: Double
        /// The painted/puckered width of one vein, blade units.
        public var width: Double
        public init(pairs: Int, from: Double, to: Double, angle: Double, sweep: Double, width: Double) {
            self.pairs = pairs; self.from = from; self.to = to
            self.angle = angle; self.sweep = sweep; self.width = width
        }

        /// 0 … 1: how near (x, y) lies to the nearest vein's line.
        public func proximity(x: Double, y: Double) -> Double {
            guard pairs > 0 else { return 0 }
            let r = abs(x)
            let rise = r / max(0.05, tan(angle)) + sweep * r * r
            var best = Double.infinity
            for k in 0 ..< pairs {
                let y0 = pairs == 1 ? from : from + (to - from) * Double(k) / Double(pairs - 1)
                best = min(best, abs(y - (y0 + rise)))
            }
            let t = best / max(1e-6, width)
            return exp(-t * t)
        }
    }

    // MARK: - A simple blade (one un-split outline)

    /// The sites of a simple blade's grid: one row per resolved margin station, and across each half
    /// the midrib, a rib-band edge a FIXED width off it (`midribBand`, so the pale rib is a crisp
    /// band whose width does not swell with the blade), the interior `columns` (fractions of the
    /// local half-width), and the margin. Rows are returned per side, midrib first.
    ///
    /// `aspect` is the blade's width over its length at `u = 1` (the `LeafSilhouette` convention).
    public static func bladeRows(margin: [LeafSilhouette.Control], aspect: Double,
                                 columns: [Double], midribBand: Double,
                                 veins: LateralVeins?) -> [[Site]] {
        let half = aspect / 2
        let inner = columns.filter { $0 > 0 && $0 < 1 }.sorted()
        return margin.map { m in
            let local = m.u * half
            let band = local > 1e-9 ? min(0.30, midribBand / local) : 0.30
            var fr: [Double] = [0, band]
            for c in inner where c > band + 0.05 { fr.append(c) }
            // Keep the column COUNT fixed row to row (a structured grid): pad with columns that
            // would have been dropped, placed evenly between the band and the margin.
            while fr.count < 2 + inner.count { fr.append(band + (1 - band) * Double(fr.count - 1) / Double(inner.count + 1)) }
            fr = Array(fr.prefix(2 + inner.count)).sorted()
            fr.append(1)
            return fr.map { f in
                let x = f * local
                return Site(x: x, y: m.v, vein: veins?.proximity(x: x, y: m.v) ?? 0)
            }
        }
    }

    /// Emit a simple blade: both halves of the `bladeRows` grid, both faces, painted.
    public static func emitBlade(into p: inout PaintedMesh, surface: Surface,
                                 margin: [LeafSilhouette.Control], aspect: Double,
                                 columns: [Double] = [0.45, 0.8], midribBand: Double = 0.016,
                                 veins: LateralVeins? = nil, paint: Paint) {
        let rows = bladeRows(margin: margin, aspect: aspect, columns: columns,
                             midribBand: midribBand, veins: veins)
        guard rows.count >= 2 else { return }
        for side in [1.0, -1.0] {
            func mirrored(_ s: Site) -> Site { Site(x: s.x * side, y: s.y, vein: s.vein) }
            for i in 0 ..< rows.count - 1 {
                let r0 = rows[i], r1 = rows[i + 1]
                for j in 0 ..< min(r0.count, r1.count) - 1 {
                    // Counter-clockwise seen from the upper face on the + half; `emitCell` rewinds
                    // each face to its own outward, so the mirrored half needs nothing special.
                    emitCell(into: &p, [mirrored(r0[j]), mirrored(r0[j + 1]),
                                        mirrored(r1[j + 1]), mirrored(r1[j])],
                             surface: surface, paint: paint)
                }
            }
        }
    }

    // MARK: - A monstera blade: a heart cut by splits and pierced by holes

    /// A monstera leaf is described in VEIN coordinates rather than by an outline. `a` indexes the
    /// fan of lateral veins round one half of the blade — 0 the basal lobe (the veins that leave the
    /// petiole junction and fan down into the heart's lobe), 1 the tip — and `r` runs out along one
    /// vein, 0 on the midrib, 1 on the margin. The half-blade is then the unit square of (a, r), a
    /// structured grid; a SPLIT is the cells `[a ± w/2] × [depth, 1]` left out, and a HOLE (a
    /// fenestration) is `[a ± w/2] × [r0, r1]` left out with tissue both inside and outside it.
    ///
    /// This is why the monstera stopped being a lettuce: its read is the cuts, and cuts are cells
    /// that are not there — nothing a Catmull-Rom margin could ever draw (DH-0775).
    public struct MonsteraCuts: Sendable, Equatable {
        public struct Split: Sendable, Equatable {
            public var a: Double, width: Double, depth: Double
            public init(a: Double, width: Double, depth: Double) { self.a = a; self.width = width; self.depth = depth }
        }
        public struct Hole: Sendable, Equatable {
            public var a: Double, width: Double, r0: Double, r1: Double
            public init(a: Double, width: Double, r0: Double, r1: Double) { self.a = a; self.width = width; self.r0 = r0; self.r1 = r1 }
        }
        public var splits: [Split]
        public var holes: [Hole]
        public init(splits: [Split], holes: [Hole]) { self.splits = splits; self.holes = holes }
    }

    /// The monstera's right-half margin, base → round the basal lobe → widest → tip, in blade units
    /// (x across, y along). An open heart: the lobes reach 10 % of the length below the petiole.
    public static let monsteraMargin: [Vec2] = [
        Vec2(0.035, -0.015), Vec2(0.10, -0.085), Vec2(0.22, -0.105), Vec2(0.34, -0.04), Vec2(0.43, 0.10),
        Vec2(0.475, 0.28), Vec2(0.47, 0.46), Vec2(0.42, 0.63), Vec2(0.31, 0.79), Vec2(0.16, 0.92), Vec2(0.0, 1.0)]

    /// The veins in [0, `monsteraLobeFan`] of `a` fan out of the petiole junction into the lobe.
    public static let monsteraLobeFan = 0.16
    /// The last lateral vein leaves the midrib this far up the blade.
    public static let monsteraMidribTop = 0.90
    /// Where the `r` rows fall: the rib band, the fenestration band (0.18–0.30), then out to the margin.
    public static let monsteraRows: [Double] = [0, 0.06, 0.18, 0.30, 0.48, 0.72, 1.0]

    /// The margin sampled densely and parameterised by arc length (0 lobe → 1 tip).
    static let monsteraMarginCurve: (points: [Vec2], t: [Double]) = {
        let pts = catmullRom(monsteraMargin, perSegment: 12)
        var d: [Double] = [0]
        for i in 1 ..< pts.count { d.append(d[i - 1] + len2(pts[i] - pts[i - 1])) }
        let total = max(1e-9, d[d.count - 1])
        return (pts, d.map { $0 / total })
    }()

    static func monsteraMarginPoint(_ a: Double) -> Vec2 {
        let (pts, t) = monsteraMarginCurve
        for i in 1 ..< pts.count where t[i] >= a {
            let f = (a - t[i - 1]) / max(1e-12, t[i] - t[i - 1])
            return pts[i - 1] + (pts[i] - pts[i - 1]) * f
        }
        return pts[pts.count - 1]
    }

    static func monsteraMidrib(_ a: Double) -> Vec2 {
        guard a > monsteraLobeFan else { return Vec2(0, 0) }
        return Vec2(0, monsteraMidribTop * pow((a - monsteraLobeFan) / (1 - monsteraLobeFan), 0.85))
    }

    /// A point of the right half at vein coordinates (a, r). Each vein bows a little toward the tip.
    public static func monsteraPoint(a: Double, r: Double) -> Vec2 {
        let m = monsteraMidrib(a), e = monsteraMarginPoint(a)
        let c = (m + e) * 0.5 + Vec2(0, 0.06 * (1 - abs(2 * a - 1)))
        let w0 = (1 - r) * (1 - r), w1 = 2 * r * (1 - r), w2 = r * r
        return m * w0 + c * w1 + e * w2
    }

    /// The (a) stations of one half: the lobe fan, then every cut's two edges, filled in so no
    /// segment of tissue is wider than `spacing`.
    static func monsteraColumns(_ cuts: MonsteraCuts, spacing: Double) -> [Double] {
        var edges: Set<Double> = [0, 0.05, 0.10, monsteraLobeFan, 1]
        for s in cuts.splits { edges.insert(s.a - s.width / 2); edges.insert(s.a + s.width / 2) }
        for h in cuts.holes { edges.insert(h.a - h.width / 2); edges.insert(h.a + h.width / 2) }
        let sorted = edges.filter { $0 >= 0 && $0 <= 1 }.sorted()
        var out: [Double] = [sorted[0]]
        for i in 1 ..< sorted.count {
            let lo = sorted[i - 1], hi = sorted[i]
            let n = max(1, Int(((hi - lo) / spacing).rounded(.up)))
            for k in 1 ... n { out.append(lo + (hi - lo) * Double(k) / Double(n)) }
        }
        return out
    }

    /// Emit one half of a monstera blade (`side` = +1 right, −1 left), both faces, painted. The site's
    /// `vein` is 1 along the middle of each tissue segment between two cuts — where the lateral vein
    /// runs — so the painter can draw it.
    public static func emitMonsteraHalf(into p: inout PaintedMesh, surface: Surface, cuts: MonsteraCuts,
                                        side: Double, spacing: Double = 0.045, paint: Paint) {
        let cols = monsteraColumns(cuts, spacing: spacing)
        let rows = monsteraRows
        // The vein proximity of a column: 1 half-way between two cuts (the lateral vein there),
        // falling off toward each cut. Cut centres sorted for the lookup.
        let cutCentres = (cuts.splits.map(\.a) + [monsteraLobeFan, 1.0]).sorted()
        func vein(_ a: Double) -> Double {
            var lo = 0.0, hi = 1.0
            for c in cutCentres { if c <= a { lo = c } else { hi = c; break } }
            let mid = (lo + hi) / 2, half = max(1e-6, (hi - lo) / 2)
            let t = (a - mid) / (half * 0.35)
            return exp(-t * t)
        }
        func site(_ a: Double, _ r: Double) -> Site {
            let q = monsteraPoint(a: a, r: r)
            return Site(x: q.x * side, y: q.y, vein: r > 0.08 ? vein(a) : 0)
        }
        for i in 0 ..< cols.count - 1 {
            let a0 = cols[i], a1 = cols[i + 1], am = (a0 + a1) / 2
            for j in 0 ..< rows.count - 1 {
                let r0 = rows[j], r1 = rows[j + 1], rm = (r0 + r1) / 2
                if cuts.splits.contains(where: { abs(am - $0.a) <= $0.width / 2 && rm >= $0.depth }) { continue }
                if cuts.holes.contains(where: { abs(am - $0.a) <= $0.width / 2 && rm >= $0.r0 && rm <= $0.r1 }) { continue }
                emitCell(into: &p, [site(a0, r0), site(a0, r1), site(a1, r1), site(a1, r0)],
                         surface: surface, paint: paint)
            }
        }
    }

    // MARK: - Small helpers

    static func len2(_ v: Vec2) -> Double { (v.x * v.x + v.y * v.y).squareRoot() }

    /// Uniform Catmull-Rom through 2D points (end points repeated), `perSegment` samples a segment.
    static func catmullRom(_ pts: [Vec2], perSegment n: Int) -> [Vec2] {
        guard pts.count >= 2 else { return pts }
        let P = [pts[0]] + pts + [pts[pts.count - 1]]
        var out: [Vec2] = []
        for i in 1 ..< P.count - 2 {
            let p0 = P[i - 1], p1 = P[i], p2 = P[i + 1], p3 = P[i + 2]
            for s in 0 ..< n {
                let t = Double(s) / Double(n), t2 = t * t, t3 = t2 * t
                let a = p1 * 2, b = (p2 - p0) * t
                let c = (p0 * 2 - p1 * 5 + p2 * 4 - p3) * t2
                let d = (p1 * 3 - p0 - p2 * 3 + p3) * t3
                out.append((a + b + c + d) * 0.5)
            }
        }
        out.append(pts[pts.count - 1])
        return out
    }
}
