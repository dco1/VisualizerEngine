//  GraphValue.swift — what flows along a graph's wires, and the small geometry the nodes share.
//
//  Guarantees: `GraphTransform` means the SAME placement for meshes and SDFs (scale, then rotate X, Y, Z in that
//  order, then translate), so a mesh graph and the equivalent SDF graph line up. An `SDFValue` carries a
//  conservative bounding box beside the field (fields have none of their own), which `sdfToMesh` meshes inside.
//  Not guaranteed: an SDF placement takes ONE uniform scale (the x component of `scale`) — distances stay exact;
//  mesh transforms accept non-uniform scale.

import Foundation
import simd

public struct GraphTransform: Hashable, Sendable {
    public var translate = Vec3.zero
    /// Euler degrees, applied X first, then Y, then Z.
    public var rotateDegrees = Vec3.zero
    public var scale = Vec3(1, 1, 1)
    public init(translate: Vec3 = .zero, rotateDegrees: Vec3 = .zero, scale: Vec3 = Vec3(1, 1, 1)) {
        self.translate = translate; self.rotateDegrees = rotateDegrees; self.scale = scale
    }
    init(_ a: GraphArgs) {
        self.init(translate: a.vec3("translate", .zero), rotateDegrees: a.vec3("rotate", .zero), scale: a.vec3("scale", Vec3(1, 1, 1)))
    }
    var rotation: simd_quatd { GraphTransform.quat(rotateDegrees) }

    static func quat(_ deg: Vec3) -> simd_quatd {
        let r = deg * (.pi / 180)
        return simd_quatd(angle: r.z, axis: Vec3(0, 0, 1)) * simd_quatd(angle: r.y, axis: Vec3(0, 1, 0)) * simd_quatd(angle: r.x, axis: Vec3(1, 0, 0))
    }
}

public struct GraphPoint: Hashable, Sendable {
    public var position: Vec3
    public var normal: Vec3
}

public struct SDFValue: Sendable {
    public var sdf: SDF
    public var lo: Vec3
    public var hi: Vec3
}

public enum GraphValue: Sendable {
    case mesh(Mesh3), sdf(SDFValue), points([GraphPoint]), scalar(Double), vec3(Vec3), transform(GraphTransform)

    public var type: SocketType {
        switch self {
        case .mesh: return .mesh
        case .sdf: return .sdf
        case .points: return .points
        case .scalar: return .scalar
        case .vec3: return .vec3
        case .transform: return .transform
        }
    }
    public var mesh: Mesh3? { if case .mesh(let m) = self { return m }; return nil }
    public var sdf: SDFValue? { if case .sdf(let s) = self { return s }; return nil }
    public var points: [GraphPoint]? { if case .points(let p) = self { return p }; return nil }
    public var scalar: Double? { if case .scalar(let s) = self { return s }; return nil }
}

// MARK: mesh placement

extension Mesh3 {
    /// `p' = q(p · scale) + t`. Normals follow by the inverse transpose; a mirroring scale flips the winding back.
    func placed(scale: Vec3, rotation q: simd_quatd, translation t: Vec3) -> Mesh3 {
        var out = self
        out.positions = positions.map { q.act($0 * scale) + t }
        out.normals = normals.map { normalize3(q.act($0 / scale)) }
        if scale.x * scale.y * scale.z < 0 {
            var i = 0
            while i + 2 < out.indices.count { out.indices.swapAt(i + 1, i + 2); i += 3 }
        }
        return out
    }
    func placed(_ t: GraphTransform) -> Mesh3 { placed(scale: t.scale, rotation: t.rotation, translation: t.translate) }
}

extension SDFValue {
    func placed(_ t: GraphTransform) -> SDFValue {
        let s = t.scale.x, q = t.rotation
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for i in 0 ..< 8 {
            let c = Vec3(i & 1 == 0 ? self.lo.x : self.hi.x, i & 2 == 0 ? self.lo.y : self.hi.y, i & 4 == 0 ? self.lo.z : self.hi.z)
            let w = q.act(c * s) + t.translate
            lo = simd_min(lo, w); hi = simd_max(hi, w)
        }
        return SDFValue(sdf: sdf.placed(SDF.Placement(translation: t.translate, rotation: q, scale: s)), lo: lo, hi: hi)
    }
}

// MARK: primitives

enum GraphPrimitives {
    static func box(size: Vec3) -> Mesh3 {
        var m = Mesh3()
        let h = size * 0.5
        func quad(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3, _ n: Vec3) { m.addQuad(a, b, c, d, outward: n) }
        let p = { (x: Double, y: Double, z: Double) in Vec3(x * h.x, y * h.y, z * h.z) }
        quad(p(1, -1, -1), p(1, 1, -1), p(1, 1, 1), p(1, -1, 1), Vec3(1, 0, 0))
        quad(p(-1, -1, -1), p(-1, -1, 1), p(-1, 1, 1), p(-1, 1, -1), Vec3(-1, 0, 0))
        quad(p(-1, 1, -1), p(-1, 1, 1), p(1, 1, 1), p(1, 1, -1), Vec3(0, 1, 0))
        quad(p(-1, -1, -1), p(1, -1, -1), p(1, -1, 1), p(-1, -1, 1), Vec3(0, -1, 0))
        quad(p(-1, -1, 1), p(1, -1, 1), p(1, 1, 1), p(-1, 1, 1), Vec3(0, 0, 1))
        quad(p(-1, -1, -1), p(-1, 1, -1), p(1, 1, -1), p(1, -1, -1), Vec3(0, 0, -1))
        return m
    }

    /// Solid of revolution about Y. `profile` runs (radius, y) from the bottom axis point, out and up, to the top axis point.
    static func lathe(_ profile: [Vec2], segments: Int) -> Mesh3 {
        var m = Mesh3()
        let n = Swift.max(3, segments)
        for (a, b) in zip(profile, profile.dropFirst()) where simd_length(b - a) > 1e-12 {
            let nrm = Vec2(b.y - a.y, -(b.x - a.x))
            for i in 0 ..< n {
                let t0 = 2 * Double.pi * Double(i) / Double(n), t1 = 2 * Double.pi * Double(i + 1) / Double(n), tm = (t0 + t1) / 2
                let out = Vec3(nrm.x * cos(tm), nrm.y, nrm.x * sin(tm))
                func pt(_ r: Vec2, _ t: Double) -> Vec3 { Vec3(r.x * cos(t), r.y, r.x * sin(t)) }
                m.addQuad(pt(a, t0), pt(a, t1), pt(b, t1), pt(b, t0), outward: out)
            }
        }
        return m.smoothed(creaseDegrees: 50)
    }

    static func cylinder(radius r: Double, height h: Double, segments: Int) -> Mesh3 {
        lathe([Vec2(0, -h / 2), Vec2(r, -h / 2), Vec2(r, h / 2), Vec2(0, h / 2)], segments: segments)
    }
    static func cone(radius r: Double, topRadius rt: Double, height h: Double, segments: Int) -> Mesh3 {
        lathe([Vec2(0, -h / 2), Vec2(r, -h / 2), Vec2(rt, h / 2), Vec2(0, h / 2)], segments: segments)
    }
    static func sphere(radius r: Double, segments: Int) -> Mesh3 {
        let rings = Swift.max(2, segments / 2)
        let prof = (0 ... rings).map { k -> Vec2 in
            let a = Double.pi * Double(k) / Double(rings)
            return Vec2(k == 0 || k == rings ? 0 : r * sin(a), -r * cos(a))
        }
        return lathe(prof, segments: segments)
    }
    static func torus(major: Double, minor: Double, segments: Int, tube: Int) -> Mesh3 {
        var m = Mesh3()
        let nu = Swift.max(3, segments), nv = Swift.max(3, tube)
        func pt(_ i: Int, _ j: Int) -> (p: Vec3, c: Vec3) {
            let u = 2 * Double.pi * Double(i % nu) / Double(nu), v = 2 * Double.pi * Double(j % nv) / Double(nv)
            let centre = Vec3(major * cos(u), 0, major * sin(u))
            return (centre + Vec3(minor * cos(v) * cos(u), minor * sin(v), minor * cos(v) * sin(u)), centre)
        }
        for i in 0 ..< nu { for j in 0 ..< nv {
            let a = pt(i, j), b = pt(i + 1, j), c = pt(i + 1, j + 1), d = pt(i, j + 1)
            m.addQuad(a.p, b.p, c.p, d.p, outward: (a.p + c.p) / 2 - (a.c + c.c) / 2)
        } }
        return m.smoothed(creaseDegrees: 50)
    }
}

// MARK: import-mesh blob (exact: raw little-endian doubles / uint32s, no float rounding)

public enum MeshBlob {
    public static func encode(_ m: Mesh3) -> Data {
        var d = Data("M3B1".utf8)
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func f64(_ v: Double) { var x = v.bitPattern.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        u32(m.positions.count); u32(m.normals.count); u32(m.uvs.count); u32(m.indices.count)
        for p in m.positions { f64(p.x); f64(p.y); f64(p.z) }
        for p in m.normals { f64(p.x); f64(p.y); f64(p.z) }
        for p in m.uvs { f64(p.x); f64(p.y) }
        for i in m.indices { u32(Int(i)) }
        return d
    }
    public static func decode(_ data: Data) -> Mesh3? {
        let b = [UInt8](data)
        guard b.count >= 20, Array(b[0 ..< 4]) == Array("M3B1".utf8) else { return nil }
        var o = 4
        func u32() -> Int { let v = (0 ..< 4).reduce(UInt32(0)) { $0 | UInt32(b[o + $1]) << (8 * UInt32($1)) }; o += 4; return Int(v) }
        func f64() -> Double { let v = (0 ..< 8).reduce(UInt64(0)) { $0 | UInt64(b[o + $1]) << (8 * UInt64($1)) }; o += 8; return Double(bitPattern: v) }
        let np = u32(), nn = u32(), nu = u32(), ni = u32()
        guard b.count == 20 + np * 24 + nn * 24 + nu * 16 + ni * 4 else { return nil }
        var m = Mesh3()
        m.positions = (0 ..< np).map { _ in Vec3(f64(), f64(), f64()) }
        m.normals = (0 ..< nn).map { _ in Vec3(f64(), f64(), f64()) }
        m.uvs = (0 ..< nu).map { _ in Vec2(f64(), f64()) }
        m.indices = (0 ..< ni).map { _ in UInt32(u32()) }
        return m
    }
}
