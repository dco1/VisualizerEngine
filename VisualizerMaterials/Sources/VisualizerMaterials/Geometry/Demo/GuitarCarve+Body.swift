//  GuitarCarve+Body.swift — steps 1–5: block → outline → chamfer → neck → contour. See GuitarCarve.swift.

import Foundation
import simd

extension GuitarCarve {

    // MARK: 1. the block

    static func makeBlock() -> Mesh3 { boxMesh(lo: blockLo, hi: blockHi) }

    // MARK: 2. the outline, as a field

    /// A vertical cylinder of radius r at (x, y), longer than the block is thick (so it is a 2D disc in the plan).
    static func disc(_ x: Double, _ y: Double, _ r: Double) -> SDF {
        SDF.capsule(from: Vec3(x, y, -0.1), to: Vec3(x, y, 0.1), radius: r)
    }

    /// The plan of a double-cutaway electric: lower bout, upper bout, two horns, a neck and a headstock, all extruded
    /// far through z. (x along the instrument, y across.)
    static func outlineSDF() -> SDF {
        let lower = disc(0.185, 0, 0.172)
        let upper = disc(0.375, 0, 0.128)
        var body = lower.union(upper, smooth: 0.08)
        // double cutaway: a horn on each side, a capsule from the shoulder forward past the neck's heel
        for s in [-1.0, 1.0] {
            body = body.union(prism(Vec3(0.34, s * 0.115, 0), Vec3(0.60, s * 0.092, 0), 0.036), smooth: 0.03)
        }
        let neck = SDF.box(size: Vec3(0.60, 0.056, 0.2)).translated(by: Vec3(0.67, 0, 0))
        let head = SDF.roundedBox(size: Vec3(0.22, 0.092, 0.2), radius: 0.012).translated(by: Vec3(0.95, 0, 0))
        let headFlare = disc(1.03, 0.0, 0.058).intersecting(SDF.box(size: Vec3(0.14, 0.2, 0.2)).translated(by: Vec3(1.03, 0, 0)))
        return body.union(neck, smooth: 0.03).union(head, smooth: 0.015).union(headFlare, smooth: 0.02)
    }

    /// A stadium (capsule in the plan) extruded through z: `disc` swept from `a` to `b`.
    static func prism(_ a: Vec3, _ b: Vec3, _ r: Double) -> SDF {
        let axis = b - a, l2 = dot3(axis, axis)
        return SDF.custom { p in
            let t = max(0, min(1, dot3(Vec3(p.x, p.y, 0) - a, axis) / l2))
            return len3(Vec3(p.x, p.y, 0) - (a + axis * t)) - r
        }
    }

    static func carveOutline() -> Mesh3 {
        let size = blockHi - blockLo
        let block = SDF.box(size: size).translated(by: (blockLo + blockHi) * 0.5)
        let carved = block.intersecting(outlineSDF())
        // sample lattice offset so no sample lands exactly on a face of the block
        let lo = Vec3(-0.0203, -0.2003, -0.0402), hi = Vec3(1.0803, 0.2003, 0.0402)
        return carved.mesh(lo: lo, hi: hi, cell: 0.005)
    }

    // MARK: 3. chamfer by inset + push/pull on the half-edge mesh

    static func roundEdges(_ m: Mesh3) -> Mesh3 {
        var he = HalfEdgeMesh(m)
        for side in [1.0, -1.0] {
            let flat = he.faces.filter { f in
                let n = he.faceNormal(f)
                return n.z * side > 0.9 && abs(he.faceCentroid(f).z - side * 0.025) < 0.002
            }
            var sel = he.selection(.face, flat)
            guard let inner = he.inset(sel, thickness: 0.008) else { continue }
            sel = inner
            he.pushPull(sel, distance: -0.004, normals: .averaged)
        }
        return he.toMesh3()
    }

    // MARK: 4. neck and headstock by robust boolean

    static func carveNeck(_ m: Mesh3) -> Mesh3 {
        let topCut = boxMesh(lo: Vec3(0.50, -0.05, neckTop), hi: Vec3(1.2, 0.05, 0.2))
        let underCut = boxMesh(lo: Vec3(0.50, -0.05, -0.2), hi: Vec3(1.2, 0.05, neckBottom))
        // the headstock leans back: a slab tilted 6° about y shaves its top toward the tip
        let tilt = boxMesh(lo: Vec3(-0.12, -0.2, 0), hi: Vec3(0.12, 0.2, 0.1)).rotatedY(6 * .pi / 180)
            .translated(by: Vec3(1.0, 0, 0.006))
        var r = m.booleanDifference(topCut)
        r = r.booleanDifference(underCut)
        return r.booleanDifference(tilt)
    }

    // MARK: 5. contour with sculpt brushes

    static func contour(_ m: Mesh3) -> Mesh3 {
        let vol = SculptVolume(mesh: m, cell: 0.005, padding: 4)
        // belly cut on the back, forearm bevel on the top's lower edge: big soft clay-removing stamps
        let belly = vol.beginStroke(SculptBrush(.draw, radius: 0.085, strength: 0.16, subtract: true))
        for t in stride(from: 0.0, through: 1.0, by: 0.2) {
            belly.stamp(position: Vec3(0.22 + 0.05 * t, -0.04 + 0.08 * t, -0.027), normal: Vec3(0, 0, -1))
        }
        belly.end()
        let arm = vol.beginStroke(SculptBrush(.draw, radius: 0.06, strength: 0.14, subtract: true))
        for t in stride(from: 0.0, through: 1.0, by: 0.125) {
            arm.stamp(position: Vec3(0.12 + 0.2 * t, -0.17 + 0.0 * t, 0.026), normal: Vec3(0, 0, 1))
        }
        arm.end()
        let fresh = vol.mesh()
        return fresh.decimated(targetTriangles: max(20_000, fresh.triangleCount / 3))
    }
}
