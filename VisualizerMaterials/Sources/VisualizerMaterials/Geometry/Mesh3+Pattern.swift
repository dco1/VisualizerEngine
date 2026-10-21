//  Mirror and array — the two operations a modeller reaches for after every "make one, repeat it".
//
//  A reflection flips handedness, so a naively mirrored mesh is INSIDE-OUT: its triangles keep the old
//  winding while their positions moved to the other side. `mirrored` swaps two indices of every triangle
//  and negates the reflected axis of every normal, so the result is wound and shaded outward — the same
//  contract as the mesh it came from (`MeshAudit` keeps policing it).

import simd

public extension Mesh3 {

    enum Axis: Sendable { case x, y, z }

    /// Reflect across the plane `axis = at` (default the origin plane). Winding and normals are corrected.
    func mirrored(_ axis: Axis, at plane: Double = 0) -> Mesh3 {
        func flip(_ v: Vec3) -> Vec3 {
            switch axis {
            case .x: return Vec3(2 * plane - v.x, v.y, v.z)
            case .y: return Vec3(v.x, 2 * plane - v.y, v.z)
            case .z: return Vec3(v.x, v.y, 2 * plane - v.z)
            }
        }
        func flipNormal(_ n: Vec3) -> Vec3 {
            switch axis {
            case .x: return Vec3(-n.x, n.y, n.z)
            case .y: return Vec3(n.x, -n.y, n.z)
            case .z: return Vec3(n.x, n.y, -n.z)
            }
        }
        var out = self
        out.positions = positions.map(flip)
        out.normals = normals.map(flipNormal)
        var t = 0
        while t + 2 < out.indices.count {
            out.indices.swapAt(t + 1, t + 2)                    // a reflection reverses the winding
            t += 3
        }
        return out
    }

    /// The same surface turned inside-out: windings reversed, normals negated. A closed solid becomes a CAVITY — what a
    /// camera inside a slot, a bore or a pocket must see (a normal solid is culled from the inside, and you would see
    /// straight through to whatever is behind). Use it for liners and interiors.
    func inverted() -> Mesh3 {
        var out = self
        out.normals = normals.map { -$0 }
        var t = 0
        while t + 2 < out.indices.count { out.indices.swapAt(t + 1, t + 2); t += 3 }
        return out
    }

    /// `count` copies (the original included) stepped by `offset` — a row of tines, balusters, shelf pins.
    func arrayed(count: Int, offset: Vec3) -> Mesh3 {
        var out = Mesh3()
        for i in 0..<Swift.max(0, count) { out.append(translated(by: offset * Double(i))) }
        return out
    }

    /// `count` copies spread evenly round the Y axis through `centre` — bolt circles, pendant arms, petals.
    /// `startAngle` is the first copy's turn (radians); copy i is turned by `startAngle + i·2π/count`.
    func radialArray(count: Int, around centre: Vec3 = .zero, startAngle: Double = 0) -> Mesh3 {
        var out = Mesh3()
        guard count > 0 else { return out }
        for i in 0..<count {
            let a = startAngle + Double(i) * 2 * .pi / Double(count)
            out.append(translated(by: -centre).rotatedY(a).translated(by: centre))
        }
        return out
    }
}
