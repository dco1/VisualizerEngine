//  SimpleDeform and LatticeDeform — space warps applied to every vertex.
//
//  SimpleDeform works about an axis through `origin` along `axis` (a = distance along it). `lower…upper` limit the
//  part of the axis that deforms (in a-units from `origin`); beyond a limit the deformation stops growing and the
//  mesh carries on rigidly (a bend continues straight, a twist keeps its final turn).
//   • bend    — the axis curls into an arc through `angle` radians over the limited length, toward `bendDirection`
//               (default: a perpendicular chosen from the axis). Length along the axis is preserved: an edge on the
//               axis keeps its length exactly; edges off it shear by their offset/radius.
//   • twist   — rotates each cross-section about the axis by angle·t (t = 0…1 over the limits): volume preserving.
//   • taper   — scales cross-sections by 1 + factor·t.
//   • stretch — lengthens the limited part of the axis by `1+factor` and squeezes the whole cross-section by
//               1/√(1+factor) (volume preserved over the limits; the rest of the mesh just thins).
//
//  LatticeDeform: a regular grid of control points over a box; the mesh follows them (trilinear, or cubic
//  Catmull–Rom with linear end extrapolation, both reproducing the identity lattice exactly). Points outside the box
//  take the nearest edge's displacement.
//
//  Guarantees: triangle list untouched, normals turned with their faces, deterministic. Not guaranteed: a mesh is
//  only as smooth as its vertex density — a twist of a 2-triangle quad shears it; subdivide first. CPU only (see
//  Modifier.swift).

import simd

public struct SimpleDeformModifier: Modifier {
    public enum Mode: Sendable, Equatable { case bend(angle: Double), twist(angle: Double), taper(factor: Double), stretch(factor: Double) }

    public var mode: Mode
    public var origin: Vec3
    public var axis: Vec3
    public var bendDirection: Vec3?
    public var lower: Double
    public var upper: Double

    public init(_ mode: Mode, origin: Vec3 = .zero, axis: Vec3 = Vec3(0, 1, 0), bendDirection: Vec3? = nil,
                lower: Double = 0, upper: Double = 1) {
        self.mode = mode; self.origin = origin; self.axis = normalize3(axis); self.bendDirection = bendDirection
        self.lower = lower; self.upper = Swift.max(upper, lower + 1e-9)
    }

    public var summary: String { "SimpleDeform(\(mode) origin:\(origin) axis:\(axis) limits:\(lower)…\(upper))" }

    public func apply(_ mesh: Mesh3) -> Mesh3 {
        guard !mesh.isEmpty else { return mesh }
        return mesh.moved(to: mesh.positions.map(warp))
    }

    func warp(_ p: Vec3) -> Vec3 {
        let f = frame(for: axis)
        let r = p - origin
        let a = dot3(r, f.w)
        let c = Swift.min(Swift.max(a, lower), upper)
        let t = (c - lower) / (upper - lower)
        let perp = r - f.w * a
        switch mode {
        case .twist(let angle):
            let th = angle * t, cs = cos(th), sn = sin(th)
            let rotated = perp * cs + cross3(f.w, perp) * sn
            return origin + f.w * a + rotated
        case .taper(let factor):
            return origin + f.w * a + perp * (1 + factor * t)
        case .stretch(let factor):
            let s = 1 + factor
            let k = 1 / s.squareRoot()                       // one squeeze for every point: exact volume over the limits
            return origin + f.w * (a + (s - 1) * (c - lower)) + perp * k
        case .bend(let angle):
            guard abs(angle) > 1e-9 else { return p }
            var v = bendDirection.map { $0 - f.w * dot3($0, f.w) } ?? f.u
            v = len3(v) > 1e-9 ? normalize3(v) : f.u
            let w2 = cross3(f.w, v)                          // third axis, unchanged by the bend
            let radius = (upper - lower) / angle
            let b = dot3(perp, v), cc = dot3(perp, w2)
            let th = (c - lower) / radius
            let centre = f.w * (lower + radius * sin(th)) + v * (radius * (1 - cos(th)))
            let tangent = f.w * cos(th) + v * sin(th)
            let normal = -f.w * sin(th) + v * cos(th)
            return origin + centre + normal * b + w2 * cc + tangent * (a - c)
        }
    }
}

public struct LatticeDeformModifier: Modifier {
    public enum Interpolation: Sendable { case trilinear, cubic }

    public var boxMin: Vec3
    public var boxMax: Vec3
    public var divisions: SIMD3<Int>             // control points per axis, each >= 2
    public var points: [Vec3]                     // x fastest, then y, then z; absolute positions
    public var interpolation: Interpolation

    public init(boxMin: Vec3, boxMax: Vec3, divisions: SIMD3<Int>, points: [Vec3]? = nil, interpolation: Interpolation = .cubic) {
        self.boxMin = boxMin; self.boxMax = boxMax
        self.divisions = SIMD3(Swift.max(2, divisions.x), Swift.max(2, divisions.y), Swift.max(2, divisions.z))
        self.interpolation = interpolation
        if let pts = points, pts.count == self.divisions.x * self.divisions.y * self.divisions.z { self.points = pts } else {
            self.points = LatticeDeformModifier.identityPoints(boxMin, boxMax, self.divisions)
        }
    }

    static func identityPoints(_ lo: Vec3, _ hi: Vec3, _ d: SIMD3<Int>) -> [Vec3] {
        var out: [Vec3] = []
        for k in 0..<d.z { for j in 0..<d.y { for i in 0..<d.x {
            out.append(lo + (hi - lo) * Vec3(Double(i) / Double(d.x - 1), Double(j) / Double(d.y - 1), Double(k) / Double(d.z - 1)))
        } } }
        return out
    }

    public func index(_ i: Int, _ j: Int, _ k: Int) -> Int { i + divisions.x * (j + divisions.y * k) }

    /// Move one control point by `delta`.
    public mutating func displace(_ i: Int, _ j: Int, _ k: Int, by delta: Vec3) { points[index(i, j, k)] += delta }

    public var summary: String { "Lattice(\(divisions) \(interpolation) box:\(boxMin)…\(boxMax) pts:\(points.hashValue))" }

    public func apply(_ mesh: Mesh3) -> Mesh3 {
        guard !mesh.isEmpty else { return mesh }
        return mesh.moved(to: mesh.positions.map(warp))
    }

    func warp(_ p: Vec3) -> Vec3 {
        let size = Vec3(Swift.max(boxMax.x - boxMin.x, 1e-12), Swift.max(boxMax.y - boxMin.y, 1e-12), Swift.max(boxMax.z - boxMin.z, 1e-12))
        let u = (p - boxMin) / size
        let uc = simd.min(simd.max(u, Vec3(repeating: 0)), Vec3(repeating: 1))
        let pc = boxMin + uc * size
        return p + (sample(uc) - pc)
    }

    func sample(_ u: Vec3) -> Vec3 {
        let g = u * Vec3(Double(divisions.x - 1), Double(divisions.y - 1), Double(divisions.z - 1))
        let i0 = Swift.min(Int(g.x), divisions.x - 2), j0 = Swift.min(Int(g.y), divisions.y - 2), k0 = Swift.min(Int(g.z), divisions.z - 2)
        let f = Vec3(g.x - Double(i0), g.y - Double(j0), g.z - Double(k0))
        switch interpolation {
        case .trilinear:
            func lerp(_ a: Vec3, _ b: Vec3, _ t: Double) -> Vec3 { a + (b - a) * t }
            func at(_ di: Int, _ dj: Int, _ dk: Int) -> Vec3 { points[index(i0 + di, j0 + dj, k0 + dk)] }
            let c00 = lerp(at(0, 0, 0), at(1, 0, 0), f.x), c10 = lerp(at(0, 1, 0), at(1, 1, 0), f.x)
            let c01 = lerp(at(0, 0, 1), at(1, 0, 1), f.x), c11 = lerp(at(0, 1, 1), at(1, 1, 1), f.x)
            return lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z)
        case .cubic:
            // Catmull–Rom along x, then y, then z; control points outside the grid are linear extrapolations.
            func cp(_ i: Int, _ j: Int, _ k: Int) -> Vec3 {
                // extrapolate each axis independently via 2*p0 - p1 at the ends
                func ext(_ n: Int, _ v: Int) -> (Int, Double, Int) {          // (clamped, extrapolation weight, neighbour)
                    if v < 0 { return (0, 1, 1) }
                    if v >= n { return (n - 1, 1, n - 2) }
                    return (v, 0, v)
                }
                let (ci, wi, ni) = ext(divisions.x, i), (cj, wj, nj) = ext(divisions.y, j), (ck, wk, nk) = ext(divisions.z, k)
                // Out-of-range along axes: p_out = p_edge + (p_edge - p_neighbour) per axis (separable, additive).
                var r = points[index(ci, cj, ck)]
                if wi > 0 { r += points[index(ci, cj, ck)] - points[index(ni, cj, ck)] }
                if wj > 0 { r += points[index(ci, cj, ck)] - points[index(ci, nj, ck)] }
                if wk > 0 { r += points[index(ci, cj, ck)] - points[index(ci, cj, nk)] }
                return r
            }
            func cr(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3, _ t: Double) -> Vec3 {
                0.5 * ((2 * b) + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * (t * t) + (3 * b - a - 3 * c + d) * (t * t * t))
            }
            func row(_ j: Int, _ k: Int) -> Vec3 {
                cr(cp(i0 - 1, j, k), cp(i0, j, k), cp(i0 + 1, j, k), cp(i0 + 2, j, k), f.x)
            }
            func plane(_ k: Int) -> Vec3 { cr(row(j0 - 1, k), row(j0, k), row(j0 + 1, k), row(j0 + 2, k), f.y) }
            return cr(plane(k0 - 1), plane(k0), plane(k0 + 1), plane(k0 + 2), f.z)
        }
    }
}
