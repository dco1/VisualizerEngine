import Foundation
import simd

// UVReport — stretch, distortion and texel density of a mesh's UVs, per UV island and overall.
//
// Guarantees: per triangle, the 2×2 Jacobian from the triangle's own plane to UV space is decomposed into
// singular values σ1 ≥ σ2; `anisotropy` = σ1/σ2 (1 = conformal, no shear/stretch; flipped/degenerate
// triangles count in `flipped` and are left out of the stretch means), `density` = √|det| = UV units per
// metre (geometric mean of σ1, σ2). Means are weighted by 3D area. `texelDensity(textureSize:)` is texels per
// metre. Islands are the pieces the UVs themselves break at (`uvBoundary` seams).
// Doesn't: detect overlaps between islands (use the island `bounds`; a packed mesh has disjoint boxes).
// CPU: one pass over triangles, an audit not a frame cost.

public struct UVIslandReport: Sendable {
    public var triangleCount: Int
    public var area3D: Double
    public var areaUV: Double
    public var bounds: (min: Vec2, max: Vec2)
    public var meanAnisotropy: Double
    public var maxAnisotropy: Double
    /// UV units per metre: area-weighted mean, smallest, largest.
    public var meanDensity: Double
    public var minDensity: Double
    public var maxDensity: Double
    public var flipped: Int
}

public struct UVReport: Sendable {
    public var islands: [UVIslandReport]
    public var meanAnisotropy: Double
    public var maxAnisotropy: Double
    public var meanDensity: Double
    /// Standard deviation of per-triangle density over its mean (0 = perfectly uniform).
    public var densityVariation: Double
    public var flipped: Int
    /// Share of the unit square the UV triangles cover (islands assumed disjoint).
    public var coverage: Double

    /// Texels per metre at a square texture of `textureSize` pixels.
    public func texelDensity(textureSize: Double) -> Double { meanDensity * textureSize }

    public init(_ mesh: Mesh3, tolerance: Double = 1e-6) {
        let topo = UVTopology(mesh)
        let isl = topo.islands(topo.seams(uvBoundaryOf: mesh, tolerance: tolerance))
        var reports: [UVIslandReport] = []
        var dens: [(d: Double, w: Double)] = []
        var totAniso = 0.0, totW = 0.0, maxA = 1.0, flips = 0, uvTotal = 0.0
        for island in isl {
            var a3 = 0.0, auv = 0.0, w = 0.0, sa = 0.0, ma = 1.0, sd = 0.0, lo = Double.infinity, hi = 0.0, fl = 0
            var blo = Vec2(repeating: .infinity), bhi = Vec2(repeating: -.infinity)
            for t in island.triangles {
                let i = (0..<3).map { Int(mesh.indices[3 * t + $0]) }
                let p = i.map { mesh.positions[$0] }
                let uv = mesh.uvs.count == mesh.positions.count ? i.map { mesh.uvs[$0] } : [Vec2](repeating: .zero, count: 3)
                for q in uv { blo = min(blo, q); bhi = max(bhi, q) }
                let e1 = p[1] - p[0], e2 = p[2] - p[0]
                let cr = cross3(e1, e2), area = len3(cr) / 2
                let uvA = cross2(uv[1] - uv[0], uv[2] - uv[0]) / 2
                a3 += area; auv += abs(uvA)
                guard area > 1e-14 else { continue }
                if uvA <= 1e-14 * max(area, 1e-12) { fl += 1; continue }
                let l1 = len3(e1), xa = e1 / l1, ya = normalize3(cross3(normalize3(cr), xa))
                let q1 = Vec2(l1, 0), q2 = Vec2(dot3(e2, xa), dot3(e2, ya))
                // J = [duv1 duv2] · [q1 q2]⁻¹
                let det = q1.x * q2.y - q2.x * q1.y
                let d1 = uv[1] - uv[0], d2 = uv[2] - uv[0]
                let jx = Vec2((d1.x * q2.y - d2.x * q1.y) / det, (-d1.x * q2.x + d2.x * q1.x) / det)
                let jy = Vec2((d1.y * q2.y - d2.y * q1.y) / det, (-d1.y * q2.x + d2.y * q1.x) / det)
                let f = jx.x * jx.x + jx.y * jx.y + jy.x * jy.x + jy.y * jy.y
                let dj = abs(jx.x * jy.y - jx.y * jy.x)
                let s1 = (sqrt(f + 2 * dj) + sqrt(max(f - 2 * dj, 0))) / 2
                let s2 = (sqrt(f + 2 * dj) - sqrt(max(f - 2 * dj, 0))) / 2
                let aniso = s2 > 1e-300 ? s1 / s2 : .infinity
                let d = dj.squareRoot()
                w += area; sa += aniso * area; ma = max(ma, aniso); sd += d * area
                lo = min(lo, d); hi = max(hi, d)
                dens.append((d, area))
            }
            let ok = w > 0
            reports.append(UVIslandReport(
                triangleCount: island.triangles.count, area3D: a3, areaUV: auv, bounds: (blo, bhi),
                meanAnisotropy: ok ? sa / w : 1, maxAnisotropy: ma, meanDensity: ok ? sd / w : 0,
                minDensity: ok ? lo : 0, maxDensity: hi, flipped: fl))
            totAniso += sa; totW += w; maxA = max(maxA, ma); flips += fl; uvTotal += auv
        }
        islands = reports
        meanAnisotropy = totW > 0 ? totAniso / totW : 1
        maxAnisotropy = maxA
        flipped = flips
        coverage = uvTotal
        let meanD = totW > 0 ? dens.reduce(0) { $0 + $1.d * $1.w } / totW : 0
        meanDensity = meanD
        let varD = totW > 0 ? dens.reduce(0) { $0 + ($1.d - meanD) * ($1.d - meanD) * $1.w } / totW : 0
        densityVariation = meanD > 0 ? varD.squareRoot() / meanD : 0
    }
}
