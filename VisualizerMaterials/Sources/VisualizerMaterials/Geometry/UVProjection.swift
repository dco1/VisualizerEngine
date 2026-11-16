import Foundation
import simd

// UVProjection — planar, box, cylinder and sphere projections written as UVs.
//
// Guarantees: every corner gets a UV (3 per triangle); cylinder and sphere wrap is fixed per triangle so
// a triangle straddling the seam is stretched across `u` beyond 1 instead of smeared back across the whole
// map (its UVs differ by less than half a turn); a sphere's pole corners take the mean longitude of the
// triangle's other corners. Vertices split where the same vertex needs two UVs (box edges, wrap seams).
// uv = rotate(raw · scale, rotationDegrees) + offset, with raw in metres (planar, box, height of a cylinder),
// turns 0…1 (cylinder/sphere u) or latitude 0…1 (sphere v). Doesn't: unfold anything (no distortion fixing).
// CPU: per-corner trig, run once on authoring, not per frame.

public struct UVProjection: Sendable {
    public enum Kind: Sendable { case planar, box, cylinder, sphere }
    public var kind: Kind
    /// The projection's centre: planar plane origin, cylinder/sphere centre, box centre.
    public var origin = Vec3.zero
    /// Planar: the plane's normal. Cylinder/sphere: the up axis. Box: the frame's +Z.
    public var axis = Vec3(0, 0, 1)
    /// Direction where u = 0 (the seam of a cylinder/sphere, planar u axis). Any non-parallel vector; nil picks one.
    public var reference: Vec3? = nil
    /// UV units per metre (planar, box, cylinder v); cylinder/sphere u multiplies one turn, sphere v one half-turn.
    public var scale = Vec2(1, 1)
    public var offset = Vec2.zero
    public var rotationDegrees = 0.0

    public init(_ kind: Kind, origin: Vec3 = .zero, axis: Vec3 = Vec3(0, 0, 1), reference: Vec3? = nil,
                scale: Vec2 = Vec2(1, 1), offset: Vec2 = .zero, rotationDegrees: Double = 0) {
        self.kind = kind; self.origin = origin; self.axis = axis; self.reference = reference
        self.scale = scale; self.offset = offset; self.rotationDegrees = rotationDegrees
    }

    /// Orthonormal (x, y, n) with n = axis and x as close to `reference` as the axis allows.
    func frame() -> (x: Vec3, y: Vec3, n: Vec3) {
        let n = normalize3(axis)
        var r = reference ?? (abs(n.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0))
        r -= n * dot3(r, n)
        if len3(r) < 1e-9 { r = abs(n.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0); r -= n * dot3(r, n) }
        let x = normalize3(r)
        return (x, cross3(n, x), n)
    }

    func finish(_ raw: Vec2) -> Vec2 {
        let s = raw * scale, a = rotationDegrees * .pi / 180, c = cos(a), sn = sin(a)
        return Vec2(c * s.x - sn * s.y, sn * s.x + c * s.y) + offset
    }

    /// Per-corner UVs (3 per triangle).
    public func cornerUVs(of mesh: Mesh3) -> [Vec2] {
        let f = frame()
        var out = [Vec2](); out.reserveCapacity(mesh.indices.count)
        for t in 0..<mesh.triangleCount {
            let p = (0..<3).map { mesh.positions[Int(mesh.indices[3 * t + $0])] - origin }
            let l = p.map { Vec3(dot3($0, f.x), dot3($0, f.y), dot3($0, f.n)) }
            var raw = [Vec2](repeating: .zero, count: 3)
            switch kind {
            case .planar:
                raw = l.map { Vec2($0.x, $0.y) }
            case .box:
                let nn = cross3(l[1] - l[0], l[2] - l[0])
                let a = abs(nn.x) >= abs(nn.y) && abs(nn.x) >= abs(nn.z) ? 0 : (abs(nn.y) >= abs(nn.z) ? 1 : 2)
                let flip: Double = nn[a] < 0 ? -1 : 1   // keep each face un-mirrored seen from outside
                raw = l.map { q in
                    switch a {
                    case 0: return Vec2(flip * q.y, q.z)
                    case 1: return Vec2(flip * q.z, q.x)
                    default: return Vec2(flip * q.x, q.y)
                    }
                }
            case .cylinder, .sphere:
                var lon = l.map { atan2($0.y, $0.x) / (2 * .pi) + 0.5 }
                let poles = kind == .sphere ? (0..<3).filter { hypot(l[$0].x, l[$0].y) < 1e-9 * max(1, len3(l[$0])) } : []
                let others = (0..<3).filter { !poles.contains($0) }
                if let lo = others.map({ lon[$0] }).min(), let hi = others.map({ lon[$0] }).max(), hi - lo > 0.5 {
                    for i in others where lon[i] < 0.5 { lon[i] += 1 }   // a triangle straddling the wrap
                }
                // a pole has no longitude of its own: it takes the mean of the triangle's other corners
                for i in poles { lon[i] = others.isEmpty ? 0.5 : others.map { lon[$0] }.reduce(0, +) / Double(others.count) }
                for i in 0..<3 {
                    if kind == .cylinder { raw[i] = Vec2(lon[i], l[i].z) } else {
                        raw[i] = Vec2(lon[i], atan2(l[i].z, hypot(l[i].x, l[i].y)) / .pi + 0.5)
                    }
                }
            }
            out += raw.map(finish)
        }
        return out
    }
}

public extension Mesh3 {
    /// The mesh with `projection`'s UVs; vertices are duplicated where one vertex needs two UVs.
    func projectedUVs(_ projection: UVProjection) -> Mesh3 {
        withCornerUVs(projection.cornerUVs(of: self))
    }
}
