import Foundation
import simd
import VisualizerMaterials

// The shared kit the three ROSETTE succulents (DH-0797 — Aloe, Echeveria, Century Plant) are built
// from. A rosette succulent's leaf is a SOLID — centimetres of water-storing tissue with a rim you can
// see the thickness of — so it goes through `PadConstructor`, the one closed-solid constructor (the
// bunny ears' pad), never a double-sided blade. The flat crowned pad is then SHAPED into a rosette
// leaf here, the way `bunnyEarsDisplace` warps a pad: it thins toward the tip, its margins lift into
// a channel, and its midrib arches. No new engine surface was needed — the pad's outline already
// carries the taper to a point.
//
// Parts, per the bridge's treatment: the fleshy leaves ride the opaque `stem` part (a thick leaf on
// the thin-sheet backlight glows like a card). What rides `foliage` is species-owned — the aloe's and
// agave's teeth and spines, the echeveria's blushed tips.

extension GardenPlantMesh {

    /// One thick fleshy rosette leaf, before it is meshed — the single source its mesh, its teeth,
    /// its spine and its tests all read.
    public struct FleshyLeaf: Sendable, Equatable {
        /// Where the leaf leaves the rosette (its pinched base is sunk inside the core).
        public var base: Vec3
        /// Base → tip, before the arch bends it.
        public var yAxis: Vec3
        /// The upper (inner) face.
        public var normal: Vec3
        public var length: Double
        /// The widest width.
        public var width: Double
        /// Rim thickness at the base; the midline is crowned `crown` × this thicker.
        public var thickness: Double
        public var crown: Double
        /// Thickness at the tip as a fraction of the base's — the leaf thins as it tapers.
        public var tipThickness: Double
        /// How far the margins lift, × width: the upper face becomes a channel, the lower convex.
        public var channel: Double
        /// The midrib's total turn (radians) AWAY from the upper face — a recurved leaf. Negative
        /// curls it toward the upper face: an incurved, cupped leaf.
        public var arch: Double

        public var xAxis: Vec3 { normalize3(cross3(yAxis, normal)) }
    }

    /// The golden angle — successive leaves of a rosette are this far apart round its axis.
    public static let goldenAngle = Double.pi * (3 - sqrt(5))

    /// Leaf `index`'s outward direction in a golden-angle rosette about `axis`.
    public static func rosetteRadial(axis: Vec3, index: Int, phase: Double) -> Vec3 {
        let f = frame(yAxis: axis)
        let az = phase + Double(index) * goldenAngle
        return normalize3(f.x * cos(az) + f.z * sin(az))
    }

    /// A rosette leaf's frame: pitched `pitch` radians up out of the rosette's plane along `radial`,
    /// its upper face looking in toward the axis (straight up when the leaf lies flat).
    public static func rosetteLeafFrame(axis: Vec3, radial: Vec3, pitch: Double) -> (yAxis: Vec3, normal: Vec3) {
        let a = normalize3(axis)
        return (normalize3(radial * cos(pitch) + a * sin(pitch)),
                normalize3(a * cos(pitch) - radial * sin(pitch)))
    }

    /// The half-width fraction of `margin` at `v` (0 base → 1 tip).
    public static func halfWidth(_ margin: [LeafSilhouette.Control], atV v: Double) -> Double {
        for i in 0 ..< max(0, margin.count - 1) where v >= margin[i].v && v <= margin[i + 1].v {
            let span = margin[i + 1].v - margin[i].v
            return span < 1e-9 ? margin[i + 1].u : margin[i].u + (margin[i + 1].u - margin[i].u) * (v - margin[i].v) / span
        }
        return 0
    }

    /// Shape a point of the flat, crowned pad into the leaf: the midrib bends on a circular arc of
    /// total turn `arch`, the thickness tapers toward the tip, and the margins lift into a channel.
    /// Every displacement moves a point along its OWN station's normal and scales its offset by a
    /// positive factor, so the two faces can never be pushed through each other.
    public static func fleshyLeafDisplace(_ l: FleshyLeaf, _ q: Vec3) -> Vec3 {
        let d = q - l.base
        let along = dot3(d, l.yAxis), across = dot3(d, l.xAxis), off = dot3(d, l.normal)
        let v = min(1, max(0, along / l.length))
        let xl = across / (l.width / 2)
        let k = l.arch / l.length
        let centre: Vec3, n: Vec3
        if abs(k) < 1e-9 {
            centre = l.base + l.yAxis * along
            n = l.normal
        } else {
            centre = l.base + l.yAxis * (sin(k * along) / k) + l.normal * ((cos(k * along) - 1) / k)
            n = l.normal * cos(k * along) + l.yAxis * sin(k * along)
        }
        let thick = off * (1 + (l.tipThickness - 1) * v)
        let trough = l.channel * l.width * xl * xl * smoothstep(0, 0.25, v)
        return centre + l.xAxis * across + n * (thick + trough)
    }

    /// A meshed leaf, and where along it (0 base → 1 tip) each triangle sits — so a species can cut
    /// the leaf into colour bands without re-deriving the geometry.
    public struct FleshyLeafBuild {
        public var mesh: Mesh3
        public var triangleV: [Double]
    }

    /// Mesh one leaf on `silhouette`'s `tier` outline (unsubdivided — the outline's own controls are
    /// the stations along the leaf, and the triangle budget).
    ///
    /// Goes through the SILHOUETTE entry point on purpose: called from here, the resolved-`margin:`
    /// overload of `PadConstructor.emitPad` returns its triangle count but leaves a `Mesh3` sink
    /// empty and corrupts memory (the next build segfaults) — found building these rosettes.
    public static func fleshyLeafBuild(_ l: FleshyLeaf, silhouette: LeafSilhouette, tier: LeafSilhouette.Tier) -> FleshyLeafBuild {
        var m = Mesh3()
        PadConstructor.emitPad(into: &m,
                               placement: .init(position: l.base + l.yAxis * (l.length * 0.46),
                                                xAxis: l.xAxis, yAxis: l.yAxis, bentNormal: l.normal,
                                                width: l.width, height: l.length),
                               silhouette: silhouette, tier: tier, subdivisions: 1,
                               thickness: l.thickness, crown: l.crown, winding: .singleSided)
        var tv: [Double] = []
        tv.reserveCapacity(m.triangleCount)
        for t in 0 ..< m.triangleCount {
            var c = Vec3(0, 0, 0)
            for k in 0 ..< 3 { c = c + m.positions[Int(m.indices[t * 3 + k])] }
            tv.append(dot3(c * (1.0 / 3.0) - l.base, l.yAxis) / l.length)
        }
        m.positions = m.positions.map { fleshyLeafDisplace(l, $0) }
        // A wide crease: the rim band blends into both faces, so the edge reads ROUNDED — plump.
        return FleshyLeafBuild(mesh: m.smoothed(creaseDegrees: 100), triangleV: tv)
    }

    /// Mesh one leaf PAINTED: `fleshyLeafBuild`'s solid, every vertex coloured by `paint` from its
    /// place on the leaf (`v` 0 base → 1 tip; `u` 0 midline → 1 margin) and its face (`.rim` is the
    /// thickness band). This is how a solid leaf wears a pattern with no colour groups — a snake
    /// plant's bands, an echeveria's blush fading in rather than cut at a line (`split`).
    ///
    /// Same silhouette entry point as `fleshyLeafBuild`, for the same reason (see its note on the
    /// resolved-`margin:` overload).
    public static func fleshyLeafPainted(_ l: FleshyLeaf, silhouette: LeafSilhouette, tier: LeafSilhouette.Tier,
                                         subdivisions: Int = 1,
                                         paint: @escaping (LeafStripVertex<Double>, LeafFace) -> Vec3) -> PaintedMesh {
        var sink = PaintingSink(upper: l.normal, paint: paint)
        PadConstructor.emitPad(into: &sink,
                               placement: .init(position: l.base + l.yAxis * (l.length * 0.46),
                                                xAxis: l.xAxis, yAxis: l.yAxis, bentNormal: l.normal,
                                                width: l.width, height: l.length),
                               silhouette: silhouette, tier: tier, subdivisions: subdivisions,
                               thickness: l.thickness, crown: l.crown, winding: .singleSided)
        var p = sink.painted
        p.mesh.positions = p.mesh.positions.map { fleshyLeafDisplace(l, $0) }
        // A wide crease, as `fleshyLeafBuild`: the rim blends into both faces — the edge reads ROUNDED.
        return p.smoothed(creaseDegrees: 100)
    }

    /// Cut a built leaf at `vCut`: the triangles nearer the base, and the ones nearer the tip. The
    /// whole leaf was smoothed as ONE part first, so the shading is continuous across the cut.
    public static func split(_ b: FleshyLeafBuild, at vCut: Double) -> (base: Mesh3, tip: Mesh3) {
        if vCut >= 1 { return (b.mesh, Mesh3()) }
        var lo: [Int] = [], hi: [Int] = []
        for (t, v) in b.triangleV.enumerated() {
            if v < vCut { lo.append(t) } else { hi.append(t) }
        }
        return (triangles(of: b.mesh, at: lo), triangles(of: b.mesh, at: hi))
    }

    /// A copy of the listed triangles of `mesh` (vertices re-indexed).
    public static func triangles(of mesh: Mesh3, at list: [Int]) -> Mesh3 {
        var out = Mesh3()
        for t in list { out.append(triangles(of: mesh, in: t ..< t + 1)) }
        return out
    }

    /// A point on the shaped leaf's UPPER face at (`v`, `u` of the local half-width, `side` ±1),
    /// lifted `lift` off it — the same crowned surface `PadConstructor` builds.
    public static func fleshyLeafUpperFace(_ l: FleshyLeaf, margin: [LeafSilhouette.Control],
                                    v: Double, u: Double, side: Double, lift: Double) -> Vec3 {
        let across = side * u * halfWidth(margin, atV: v) * l.width / 2
        let off = l.thickness / 2 + l.crown * l.thickness * (1 - u * u) * sin(v * Double.pi) + lift
        return fleshyLeafDisplace(l, l.base + l.yAxis * (l.length * v) + l.xAxis * across + l.normal * off)
    }

    /// The shaped leaf's margin at `v` on `side`: the rim's mid-thickness point, the outward
    /// direction across the leaf there, and the direction toward the tip.
    public static func fleshyLeafMargin(_ l: FleshyLeaf, margin: [LeafSilhouette.Control],
                                 v: Double, side: Double) -> (point: Vec3, out: Vec3, forward: Vec3) {
        let across = side * halfWidth(margin, atV: v) * l.width / 2
        func at(_ vv: Double, _ a: Double) -> Vec3 {
            fleshyLeafDisplace(l, l.base + l.yAxis * (l.length * vv) + l.xAxis * a)
        }
        let p = at(v, across)
        let forward = normalize3(at(min(1, v + 0.01), across) - at(max(0, v - 0.01), across))
        let out = normalize3(at(v, across + side * l.width * 0.02) - p)
        return (p, out, forward)
    }

    /// Marginal teeth: `perSide` on each margin between `v0` and `v1`, each a small cone standing
    /// out of the rim and hooked `hook` of the way toward the tip. Its foot is sunk into the rim and
    /// never wider than the rim is thick there, so it cannot show a flange on either face. Three
    /// triangles a tooth — the teeth are merged low-poly geometry, never per-tooth solids.
    public static func appendMarginTeeth(_ m: inout Mesh3, leaf l: FleshyLeaf, margin: [LeafSilhouette.Control],
                                  perSide: Int, from v0: Double, to v1: Double,
                                  length: Double, hook: Double, stoutness: Double) {
        guard perSide > 0 else { return }
        for side in [1.0, -1.0] {
            for i in 0 ..< perSide {
                let v = v0 + (v1 - v0) * (Double(i) + 0.5) / Double(perSide)
                let r = fleshyLeafMargin(l, margin: margin, v: v, side: side)
                let len = length * (1 - 0.45 * v)
                let rim = l.thickness * (1 + (l.tipThickness - 1) * v)
                let dir = normalize3(r.out + r.forward * hook)
                let foot = r.point - r.out * (len * 0.3)
                appendNeedle(&m, base: foot, tip: foot + dir * len, radius: min(len * stoutness, rim * 0.45))
            }
        }
    }

    /// The leaf's tip as a point plus the direction it points, on the shaped leaf.
    public static func fleshyLeafTip(_ l: FleshyLeaf) -> (point: Vec3, direction: Vec3) {
        let tip = fleshyLeafDisplace(l, l.base + l.yAxis * l.length)
        let before = fleshyLeafDisplace(l, l.base + l.yAxis * (l.length * 0.97))
        return (tip, normalize3(tip - before))
    }
}
