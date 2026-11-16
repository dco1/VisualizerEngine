//  SDF.swift — signed-distance-field modelling (ticket item C, "Modelling kit").
//
//  An `SDF` is an immutable, `Sendable` TREE of primitives and operators; `sdf(p)` is the signed distance
//  (negative inside, in metres) at a point. Trees, not bare closures, so the same model can be evaluated on the CPU
//  (Double, here) or compiled to a flat program for Metal (`SDF+GPU.swift`, Float).
//
//  Guarantees: every primitive is an exact (or conservative) Euclidean distance; hard booleans, offset and shell keep
//  that (up to the usual min/max under-estimate away from the surface); smooth booleans are polynomial blends (iq).
//  Not guaranteed: twist / bend and non-uniform results are only Lipschitz-ish (the surface is exact, the distance
//  away from it is not), and `.custom` / `.mesh` / twist / bend have no GPU form.
//  Meshing lives in `SDF+SurfaceNets.swift`; mesh-to-SDF in `SDFMeshField.swift`.

import simd

public struct SDF: Sendable {
    /// Where a `shell` puts its wall relative to the original surface.
    public enum ShellSide: UInt32, Sendable { case inside = 0, centered = 1, outside = 2 }

    /// A rigid-ish placement: `p' = rotation⁻¹ · (p − translation) / scale` (uniform scale only, so distances stay exact).
    public struct Placement: Sendable {
        public var translation: Vec3 = .zero
        public var rotation = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
        public var scale: Double = 1
        public init(translation: Vec3 = .zero, rotation: simd_quatd = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1), scale: Double = 1) {
            self.translation = translation; self.rotation = rotation; self.scale = scale
        }
        @inlinable func toLocal(_ p: Vec3) -> Vec3 { rotation.inverse.act(p - translation) / scale }
    }

    public indirect enum Node: Sendable {
        case sphere(radius: Double)
        case box(half: Vec3)
        case roundedBox(half: Vec3, radius: Double)      // `half` is the outer half extent
        case capsule(a: Vec3, b: Vec3, radius: Double)
        case torus(major: Double, minor: Double)         // in the XZ plane
        case cone(halfHeight: Double, bottomRadius: Double, topRadius: Double)   // along Y, centred; a radius may be 0
        case plane(normal: Vec3, offset: Double)         // inside is below: d = n·p − offset
        case mesh(SDFMeshField)
        case custom(@Sendable (Vec3) -> Double)
        case place(Placement, Node)
        case twist(rate: Double, Node)                   // radians per metre of Y
        case bend(rate: Double, Node)                    // radians per metre of X
        case union(Node, Node, k: Double)
        case subtract(Node, Node, k: Double)             // a minus b
        case intersect(Node, Node, k: Double)
        case negate(Node)
        case offset(Node, by: Double)
        case shell(Node, thickness: Double, side: ShellSide)
    }

    public let node: Node
    public init(_ node: Node) { self.node = node }

    // MARK: evaluation

    public func callAsFunction(_ p: Vec3) -> Double { SDF.eval(node, p) }
    public func distance(at p: Vec3) -> Double { SDF.eval(node, p) }

    /// Central-difference gradient (unnormalised) with half-step `h`.
    public func gradient(at p: Vec3, h: Double = 1e-4) -> Vec3 {
        Vec3(SDF.eval(node, p + Vec3(h, 0, 0)) - SDF.eval(node, p - Vec3(h, 0, 0)),
             SDF.eval(node, p + Vec3(0, h, 0)) - SDF.eval(node, p - Vec3(0, h, 0)),
             SDF.eval(node, p + Vec3(0, 0, h)) - SDF.eval(node, p - Vec3(0, 0, h))) / (2 * h)
    }

    static func smin(_ a: Double, _ b: Double, _ k: Double) -> Double {
        guard k > 0 else { return Swift.min(a, b) }
        let h = Swift.max(k - abs(a - b), 0) / k
        return Swift.min(a, b) - h * h * k * 0.25
    }
    static func smax(_ a: Double, _ b: Double, _ k: Double) -> Double { -smin(-a, -b, k) }

    static func eval(_ n: Node, _ p: Vec3) -> Double {
        switch n {
        case .sphere(let r): return len3(p) - r
        case .box(let h):
            let q = simd_abs(p) - h
            return len3(simd_max(q, .zero)) + Swift.min(Swift.max(q.x, Swift.max(q.y, q.z)), 0)
        case .roundedBox(let h, let r):
            let q = simd_abs(p) - (h - Vec3(repeating: r))
            return len3(simd_max(q, .zero)) + Swift.min(Swift.max(q.x, Swift.max(q.y, q.z)), 0) - r
        case .capsule(let a, let b, let r):
            let pa = p - a, ba = b - a
            let t = Swift.max(0, Swift.min(1, dot3(pa, ba) / Swift.max(dot3(ba, ba), 1e-30)))
            return len3(pa - ba * t) - r
        case .torus(let big, let small):
            let q = SIMD2(len3(Vec3(p.x, 0, p.z)) - big, p.y)
            return (q.x * q.x + q.y * q.y).squareRoot() - small
        case .cone(let h, let r1, let r2):
            let q = SIMD2((p.x * p.x + p.z * p.z).squareRoot(), p.y)
            let k1 = SIMD2(r2, h), k2 = SIMD2(r2 - r1, 2 * h)
            let ca = SIMD2(q.x - Swift.min(q.x, q.y < 0 ? r1 : r2), abs(q.y) - h)
            let t = Swift.max(0, Swift.min(1, simd_dot(k1 - q, k2) / simd_dot(k2, k2)))
            let cb = q - k1 + k2 * t
            let s: Double = (cb.x < 0 && ca.y < 0) ? -1 : 1
            return s * Swift.min(simd_dot(ca, ca), simd_dot(cb, cb)).squareRoot()
        case .plane(let nrm, let off): return dot3(p, nrm) - off
        case .mesh(let m): return m.signedDistance(p)
        case .custom(let f): return f(p)
        case .place(let pl, let c): return eval(c, pl.toLocal(p)) * pl.scale
        case .twist(let k, let c):
            let a = k * p.y, cs = cos(a), sn = sin(a)
            return eval(c, Vec3(cs * p.x - sn * p.z, p.y, sn * p.x + cs * p.z))
        case .bend(let k, let c):
            let a = k * p.x, cs = cos(a), sn = sin(a)
            return eval(c, Vec3(cs * p.x - sn * p.y, sn * p.x + cs * p.y, p.z))
        case .union(let a, let b, let k): return smin(eval(a, p), eval(b, p), k)
        case .subtract(let a, let b, let k): return smax(eval(a, p), -eval(b, p), k)
        case .intersect(let a, let b, let k): return smax(eval(a, p), eval(b, p), k)
        case .negate(let c): return -eval(c, p)
        case .offset(let c, let r): return eval(c, p) - r
        case .shell(let c, let t, let side):
            let d = eval(c, p)
            switch side {
            case .inside: return Swift.max(d, -(d + t))
            case .centered: return abs(d) - t * 0.5
            case .outside: return Swift.max(d - t, -d)
            }
        }
    }
}

// MARK: primitives

public extension SDF {
    static func sphere(radius: Double) -> SDF { SDF(.sphere(radius: radius)) }
    static func box(size: Vec3) -> SDF { SDF(.box(half: size * 0.5)) }
    static func roundedBox(size: Vec3, radius: Double) -> SDF { SDF(.roundedBox(half: size * 0.5, radius: radius)) }
    static func capsule(from a: Vec3, to b: Vec3, radius: Double) -> SDF { SDF(.capsule(a: a, b: b, radius: radius)) }
    static func torus(major: Double, minor: Double) -> SDF { SDF(.torus(major: major, minor: minor)) }
    /// A cone frustum along Y centred on the origin; `topRadius: 0` is a true cone (apex up).
    static func cone(height: Double, bottomRadius: Double, topRadius: Double = 0) -> SDF {
        SDF(.cone(halfHeight: height * 0.5, bottomRadius: bottomRadius, topRadius: topRadius))
    }
    static func plane(normal: Vec3, offset: Double = 0) -> SDF { SDF(.plane(normal: normalize3(normal), offset: offset)) }
    /// A closed mesh as a field (`SDFMeshField`: BVH closest point, sign by angle-weighted pseudo-normals).
    static func mesh(_ m: Mesh3) -> SDF { SDF(.mesh(SDFMeshField(m))) }
    static func custom(_ f: @escaping @Sendable (Vec3) -> Double) -> SDF { SDF(.custom(f)) }
}

// MARK: transforms

public extension SDF {
    func translated(by t: Vec3) -> SDF { SDF(.place(Placement(translation: t), node)) }
    func rotated(by q: simd_quatd) -> SDF { SDF(.place(Placement(rotation: q), node)) }
    func rotated(angle: Double, axis: Vec3) -> SDF { rotated(by: simd_quatd(angle: angle, axis: normalize3(axis))) }
    /// Uniform scale about the origin (distances stay exact).
    func scaled(by s: Double) -> SDF { SDF(.place(Placement(scale: s), node)) }
    func placed(_ p: Placement) -> SDF { SDF(.place(p, node)) }
    /// Twist about Y by `rate` radians per metre. Not distance-exact away from the surface; CPU only.
    func twisted(rate: Double) -> SDF { SDF(.twist(rate: rate, node)) }
    /// Bend about Z by `rate` radians per metre of X. Not distance-exact away from the surface; CPU only.
    func bent(rate: Double) -> SDF { SDF(.bend(rate: rate, node)) }
}

// MARK: booleans and surface operators

public extension SDF {
    /// `smooth` is the blend width `k` in metres (0 = hard).
    func union(_ other: SDF, smooth k: Double = 0) -> SDF { SDF(.union(node, other.node, k: k)) }
    func subtracting(_ other: SDF, smooth k: Double = 0) -> SDF { SDF(.subtract(node, other.node, k: k)) }
    func intersecting(_ other: SDF, smooth k: Double = 0) -> SDF { SDF(.intersect(node, other.node, k: k)) }
    func inverted() -> SDF { SDF(.negate(node)) }

    /// Grow (`r` > 0) or shrink (`r` < 0) the solid by `r` metres.
    func offset(by r: Double) -> SDF { SDF(.offset(node, by: r)) }

    /// Hollow the solid to a wall `thickness` metres thick (default: inward from the surface, so the outside is unchanged).
    func shell(thickness: Double, side: ShellSide = .inside) -> SDF { SDF(.shell(node, thickness: thickness, side: side)) }

    /// Cut `tool` into this solid, no deeper than `depth` metres below the surface (an engraving / dent, not a through
    /// cut), with the edges eased over `falloff` metres. It is `self − (tool ∩ {points within depth of the surface})`,
    /// so it is plain tree algebra and runs on the GPU too.
    func carved(tool: SDF, depth: Double, falloff: Double = 0) -> SDF {
        let skin = SDF(.negate(.offset(node, by: -depth)))      // d_skin = −d − depth: negative within `depth` of the surface
        return subtracting(tool.intersecting(skin, smooth: falloff), smooth: falloff)
    }
}
