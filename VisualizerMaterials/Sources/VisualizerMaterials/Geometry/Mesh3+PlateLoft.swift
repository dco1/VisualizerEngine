//  Plate loft — a thin ware (fork, knife, spatula, hinge leaf, key) as ONE closed solid with real
//  rounded edges, swept from sections.
//
//  The older flatware builder lofted a rectangle per station: top row, bottom row, and a flat
//  vertical wall joining them. Every lengthwise edge was therefore a 90° arris — a paper-sharp
//  line that catches no highlight, which is what reads as "CG" on metal. Here each station is a
//  rounded section (a stadium-ended slab): the top surface, a quarter-round of `edgeRadius`, the
//  wall, a quarter-round and the bottom surface, so the ROUNDING IS IN THE GEOMETRY and the
//  silhouette is soft, not just the shading.
//
//  The caller owns the shape: a station is a z, a row of lateral x samples, the bottom surface's
//  height at each, and a thickness. Because the caller evaluates one global surface function at
//  every sample, two lofts that meet (a fork's head and its tines) land on the SAME surface and
//  the seam cannot step.
//
//  Winding is declared through `addQuad(outward:)` from the section's own outward normal, so
//  `MeshAudit` keeps policing it. Run the result through `smoothed(creaseDegrees:)` (≥ the arc's
//  step angle, < 90°) before use: the quarter-rounds then shade as curves and the end caps stay
//  creased.

import simd

public struct PlateStation: Sendable {
    public var z: Double
    /// Lateral samples, left → right (ascending, ≥ 2). They may collapse to one x (a pointed end).
    public var x: [Double]
    /// Height of the BOTTOM surface at each lateral sample.
    public var yBottom: [Double]
    /// Slab thickness at this station; the top surface is `yBottom + thickness`.
    public var thickness: Double

    public init(z: Double, x: [Double], yBottom: [Double], thickness: Double) {
        self.z = z; self.x = x; self.yBottom = yBottom; self.thickness = thickness
    }
}

public extension Mesh3 {

    /// Sweep `stations` (ordered along z) into a closed solid whose lengthwise edges are rounded to
    /// `edgeRadius` (clamped to 45 % of the local thickness and width, so a thin or pointed end
    /// closes instead of inverting). `edgeSegments` facets per quarter-round.
    static func platedLoft(_ stations: [PlateStation], edgeRadius: Double,
                           edgeSegments: Int = 3) -> Mesh3 {
        guard stations.count >= 2, stations.allSatisfy({ $0.x.count >= 2 && $0.x.count == $0.yBottom.count })
        else { return Mesh3() }
        let lateral = stations[0].x.count
        guard stations.allSatisfy({ $0.x.count == lateral }) else { return Mesh3() }
        let segs = Swift.max(1, edgeSegments)
        let loops = stations.map { section($0, edgeRadius: edgeRadius, segments: segs) }
        let m = loops[0].count
        var mesh = Mesh3()

        for i in 0..<(stations.count - 1) {
            let a = loops[i], b = loops[i + 1]
            for k in 0..<m {
                let k1 = (k + 1) % m
                // Outward = the section's own outward normal, from whichever station has a tangent.
                var d = a[k1] - a[k]
                if len3(d) < 1e-12 { d = b[k1] - b[k] }
                guard len3(d) > 1e-12 else { continue }
                let outward = Vec3(-d.y, d.x, 0)
                mesh.addQuad(a[k], a[k1], b[k1], b[k], outward: outward)
            }
        }
        for (loop, dir) in [(loops[0], -1.0), (loops[stations.count - 1], 1.0)] {
            let c = loop.reduce(Vec3.zero) { $0 + $1 } / Double(loop.count)
            for k in 0..<m {
                mesh.addTriangle(c, loop[k], loop[(k + 1) % m], outward: Vec3(0, 0, dir))
            }
        }
        return mesh
    }

    private static func section(_ s: PlateStation, edgeRadius: Double, segments: Int) -> [Vec3] {
        let n = s.x.count - 1
        let x0 = s.x[0], xn = s.x[n], t = s.thickness
        let width = xn - x0
        let r = width > 1e-9 ? Swift.max(0, Swift.min(edgeRadius, 0.45 * t, 0.45 * width)) : 0
        let z = s.z
        func p(_ x: Double, _ y: Double) -> Vec3 { Vec3(x, y, z) }
        func inner(_ j: Int) -> Double {
            width > 1e-9 ? x0 + r + (xn - x0 - 2 * r) * (s.x[j] - x0) / width : x0
        }
        func arc(cx: Double, cy: Double, from a: Double, to b: Double, _ k: Int) -> Vec3 {
            let th = (a + (b - a) * Double(k) / Double(segments)) * .pi / 180
            return p(cx + r * cos(th), cy + r * sin(th))
        }
        let yL = s.yBottom[0], yR = s.yBottom[n]
        var loop: [Vec3] = []
        for j in 0...n { loop.append(p(inner(j), s.yBottom[j] + t)) }                         // top, left → right
        for k in 1...segments { loop.append(arc(cx: xn - r, cy: yR + t - r, from: 90, to: 0, k)) }     // top-right round
        for k in 0...segments { loop.append(arc(cx: xn - r, cy: yR + r, from: 0, to: -90, k)) }        // bottom-right round
        for j in stride(from: n, through: 0, by: -1) { loop.append(p(inner(j), s.yBottom[j])) }        // bottom, right → left
        for k in 0...segments { loop.append(arc(cx: x0 + r, cy: yL + r, from: -90, to: -180, k)) }     // bottom-left round
        for k in 0..<segments { loop.append(arc(cx: x0 + r, cy: yL + t - r, from: 180, to: 90, k)) }   // top-left round
        return loop
    }
}
