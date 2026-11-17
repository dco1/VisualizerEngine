import Foundation
import simd

// plausibility: real — helpers that read the finished shell (ray probes against the real triangles) so every class cue
// is placed ON the actual surface; stamps are closed boxes fitted to the local tangent plane. No CSG here.

/// Ray probe over one or more meshes. Triangles are bucketed along z (0.1 m) so the horizontal / vertical rays the
/// class cues use stay cheap even on 10k-triangle shells.
struct CCProbe {
    struct Tri { var a: Vec3, b: Vec3, c: Vec3, n: Vec3, lo: Vec3, hi: Vec3 }
    private(set) var tris: [Tri] = []
    private var buckets: [Int: [Int]] = [:]
    private static let cell = 0.1

    init(_ meshes: Mesh3...) { for m in meshes { add(m) } }
    init(list: [Mesh3]) { for m in list { add(m) } }

    mutating func add(_ m: Mesh3) {
        var i = 0
        while i + 2 < m.indices.count {
            let a = m.positions[Int(m.indices[i])], b = m.positions[Int(m.indices[i + 1])], c = m.positions[Int(m.indices[i + 2])]
            i += 3
            let cr = cross3(b - a, c - a)
            let l = len3(cr)
            if l < 1e-12 { continue }
            let lo = simd_min(a, simd_min(b, c)), hi = simd_max(a, simd_max(b, c))
            let idx = tris.count
            tris.append(Tri(a: a, b: b, c: c, n: cr / l, lo: lo, hi: hi))
            let k0 = Int((lo.z / Self.cell).rounded(.down)), k1 = Int((hi.z / Self.cell).rounded(.down))
            for k in k0...k1 { buckets[k, default: []].append(idx) }
        }
    }

    /// All intersections of the ray `o + t d` with t > 0, sorted by t. Normals are the triangles' face normals.
    func hits(_ o: Vec3, _ d: Vec3) -> [(t: Double, n: Vec3, p: Vec3)] {
        var out: [(t: Double, n: Vec3, p: Vec3)] = []
        func test(_ t: Tri) {
            let e1 = t.b - t.a, e2 = t.c - t.a
            let pv = cross3(d, e2)
            let det = dot3(e1, pv)
            if abs(det) < 1e-14 { return }
            let inv = 1 / det
            let tv = o - t.a
            let u = dot3(tv, pv) * inv
            if u < -1e-9 || u > 1 + 1e-9 { return }
            let qv = cross3(tv, e1)
            let v = dot3(d, qv) * inv
            if v < -1e-9 || u + v > 1 + 1e-9 { return }
            let s = dot3(e2, qv) * inv
            if s > 1e-9 { out.append((s, t.n, o + d * s)) }
        }
        if abs(d.z) < 1e-12 {
            let k = Int((o.z / Self.cell).rounded(.down))
            for k2 in [k] { for i in buckets[k2] ?? [] { test(tris[i]) } }
        } else {
            for t in tris { test(t) }
        }
        out.sort { $0.t < $1.t }
        return out
    }

    /// Highest surface y above/below the vertical line through (x, z) (nil if none).
    func top(_ x: Double, _ z: Double) -> Double? {
        let h = hits(Vec3(x, 5, z), Vec3(0, -1, 0))
        return h.first.map { 5 - $0.t }
    }

    /// Outermost surface met travelling +x (sgn +1) or -x (sgn -1) from the centre plane at height y, depth z.
    func flank(_ y: Double, _ z: Double, sign: Double = 1) -> (x: Double, n: Vec3)? {
        let h = hits(Vec3(0, y, z), Vec3(sign, 0, 0))
        return h.last.map { (sign * $0.t, $0.n) }
    }
}

extension CarContext {
    /// Body class used by the cues. Uses the Body step's exact blueprint match (`shellClass`) rather than `lampClass`,
    /// whose threshold heuristics read a modern coupe as a sedan; the taxi shares the sedan shell.
    var ccClass: LampClass {
        switch shellClass {
        case .sedan, .taxi: return .sedan
        case .hatchback: return .hatch
        case .suv: return .suv
        case .pickup: return .pickup
        case .van: return .van
        case .coupe: return .coupe
        case .bus: return .bus
        }
    }
    var ccIsTaxi: Bool { bp.roofSign }

    /// The context `CarFactory.build` makes for `spec` (used by the tests, which need rearZ / axles / cabin numbers).
    static func classCuesContext(_ spec: CarSpec) -> CarContext {
        var b = CarBlueprint.make(spec.style, era: spec.era)
        switch spec.era {
        case .classic1950s: b.cornerRadius *= 1.6; b.cabinWidthFrac -= 0.04
        case .nineties:     break
        case .modern:       b.cornerRadius *= 0.85
        }
        let L = spec.length ?? b.length, W = spec.width ?? b.width, H = spec.height ?? b.height
        let R = spec.wheelRadius ?? b.wheelRadius
        return CarContext(bp: b, era: spec.era, L: L, W: W, H: H, R: R, rL: L / b.length, rH: H / b.height)
    }

    /// A fresh accumulator with every earlier build step run (Body → Greenhouse → Lamps → BodyDetails → Wheels),
    /// i.e. exactly what `buildClassCues` sees in `CarFactory.build`. Used by the tests.
    static func classCuesBase(_ spec: CarSpec) -> (ctx: CarContext, acc: CarAccumulator) {
        let c = classCuesContext(spec)
        var a = CarAccumulator()
        c.buildBody(&a); c.buildGreenhouse(&a); c.buildLamps(&a); c.buildBodyDetails(&a); c.buildWheels(&a)
        return (c, a)
    }

    /// A thin closed box laid on the surface `probe` finds along the ray `from + t·dir` (outermost hit): `w` along the
    /// surface tangent perpendicular to `up`, `h` along `up`, sunk `back` into the paint and proud by `proud` over the
    /// local high point (fitted at the centre and four corners). `nil` if the surface is missing or too creased.
    func ccStamp(_ probe: CCProbe, from: Vec3, dir: Vec3, up: Vec3 = Vec3(0, 1, 0), w: Double, h: Double,
                 back: Double = 0.004, proud: Double = 0.0015, maxDev: Double = 0.012) -> Mesh3? {
        guard let hit = probe.hits(from, dir).last else { return nil }
        let n = normalize3(hit.n)
        let v = normalize3(up - n * dot3(up, n))
        let u = cross3(v, n)
        var lo = 0.0, hi = 0.0
        for (du, dv) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            let o = from + u * (du * w) + v * (dv * h)
            guard let hc = probe.hits(o, dir).last else { return nil }
            let dev = dot3(hc.p - hit.p, n)
            lo = min(lo, dev); hi = max(hi, dev)
        }
        if hi - lo > maxDev { return nil }
        let box = carBox(Vec3(-w / 2, -h / 2, lo - back), Vec3(w / 2, h / 2, hi + proud))
        return box.placed(xAxis: u, yAxis: v, zAxis: n, origin: hit.p)
    }

    /// A narrow strip (shut line / rail / seam) from `a` to `b` laid on the surface met along `dir` from points between
    /// them (both endpoints are probe origins inside the solid). Segments overlap by 2 mm so no two share an edge.
    func ccSeam(_ probe: CCProbe, from a: Vec3, to b: Vec3, dir: Vec3, width: Double, proud: Double = 0.0012,
                back: Double = 0.004, maxDev: Double = 0.03, step: Double = 0.10) -> Mesh3 {
        var out = Mesh3()
        let l = len3(b - a)
        guard l > 1e-4 else { return out }
        let up = (b - a) / l
        let n = max(1, Int((l / step).rounded(.up)))
        for i in 0..<n {
            let p0 = a + (b - a) * (Double(i) / Double(n)), p1 = a + (b - a) * (Double(i + 1) / Double(n))
            if let m = ccStamp(probe, from: (p0 + p1) / 2, dir: dir, up: up, w: width, h: len3(p1 - p0) + 0.002,
                               back: back, proud: proud, maxDev: maxDev) { out.append(m) }
        }
        return out
    }

    /// Prism of a convex (z, y) side-view polygon extruded across x ∈ [x0, x1].
    func ccSidePrism(_ poly: [Vec2], x0: Double, x1: Double) -> Mesh3 {
        GHKit.prism(bottom: poly.map { Vec3(x0, $0.y, $0.x) }, top: poly.map { Vec3(x1, $0.y, $0.x) })
    }

    /// Rounded-rectangle section (width × height) swept along `path`, capped; section axes follow `Mesh3.loft`.
    func ccBar(_ path: [Vec3], w: Double, h: Double, corner: Double = 0.008) -> Mesh3 {
        let prof = Mesh3.roundedRectProfile(halfWidth: w / 2, halfHeight: h / 2, cornerRadius: min(corner, h * 0.45, w * 0.45), corner: 3)
        return Mesh3.sweep(profile: prof, along: path)
    }

    /// Makes a (nearly) symmetric solid exactly mirror-symmetric: triangles are clipped at the x = 0 plane, the +x half is
    /// kept and its mirror image is appended, so every vertex has an exact mirror twin and the seam edges pair up.
    /// (A BSP boolean splits polygons in tree order, which leaves left/right vertex sets that differ by T-junction points.)
    func ccMirrorSymmetrized(_ m: Mesh3) -> Mesh3 {
        var half = Mesh3()
        typealias V = (p: Vec3, n: Vec3, uv: Vec2)
        func emit(_ poly: [V]) {
            guard poly.count >= 3 else { return }
            for i in 1..<(poly.count - 1) {
                let t = [poly[0], poly[i], poly[i + 1]]
                guard len3(cross3(t[1].p - t[0].p, t[2].p - t[0].p)) > 1e-14 else { continue }
                let base = UInt32(half.positions.count)
                for v in t { half.positions.append(v.p); half.normals.append(v.n); half.uvs.append(v.uv) }
                half.indices.append(contentsOf: [base, base + 1, base + 2])
            }
        }
        var i = 0
        while i + 2 < m.indices.count {
            let t: [V] = (0..<3).map { k in let v = Int(m.indices[i + k]); return (m.positions[v], m.normals[v], m.uvs[v]) }
            i += 3
            if t.allSatisfy({ $0.p.x >= -1e-9 }) { emit(t); continue }
            if t.allSatisfy({ $0.p.x <= 1e-9 }) { continue }
            // Sutherland–Hodgman against x >= 0
            var out: [V] = []
            for k in 0..<3 {
                let a = t[k], b = t[(k + 1) % 3]
                let ina = a.p.x >= 0, inb = b.p.x >= 0
                if ina { out.append(a) }
                if ina != inb {
                    let s = a.p.x / (a.p.x - b.p.x)
                    var p = a.p + (b.p - a.p) * s; p.x = 0
                    let n = a.n + (b.n - a.n) * s
                    out.append((p, len3(n) > 1e-12 ? normalize3(n) : a.n, a.uv + (b.uv - a.uv) * s))
                }
            }
            emit(out)
        }
        var res = half
        res.append(half.mirrored(.x))
        return res
    }

    /// Splits a triangle soup's closed solids out of `m` that lie wholly inside the box (by centroid of every triangle).
    func ccRemove(_ m: Mesh3, inside lo: Vec3, _ hi: Vec3) -> Mesh3 {
        m.partitioned { _, c in !(c.x >= lo.x && c.x <= hi.x && c.y >= lo.y && c.y <= hi.y && c.z >= lo.z && c.z <= hi.z) }.match
    }
}
