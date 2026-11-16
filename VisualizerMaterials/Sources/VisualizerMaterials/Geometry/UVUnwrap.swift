import Foundation
import simd

// UVUnwrap — least-squares conformal maps (Lévy et al. 2002) of each island, then area-normalised.
//
// Guarantees: each island is flattened with the two pinned vertices placed at the island's farthest pair
// (3D distance apart); the solve is conjugate gradients on the normal equations, written here, matrix-free
// (rows are stored per triangle), Jacobi-preconditioned and started from a planar projection. A
// developable island (a cylinder side, a cone, a flat patch) comes out as an isometry. Afterwards the island
// is scaled so its UV area equals its 3D area (1 UV unit = 1 metre, uniform texel density between islands),
// kept un-mirrored, rotated to its minimum bounding rectangle (`alignToAxes`) and moved to the origin.
// Doesn't: cut the surface (give it seams that make each island a disc; an island with a hole or handle
// folds), or avoid overlaps in a strongly curved island (LSCM is conformal, not injective).
// CPU: a sparse iterative solve per island with data-dependent convergence, run when authoring.

public struct UVUnwrapOptions: Sendable {
    public var alignToAxes = true
    /// Relative residual to stop at.
    public var tolerance = 1e-10
    public var maxIterations: Int? = nil
    public init(alignToAxes: Bool = true, tolerance: Double = 1e-10, maxIterations: Int? = nil) {
        self.alignToAxes = alignToAxes; self.tolerance = tolerance; self.maxIterations = maxIterations
    }
}

public enum UVUnwrap {
    /// Flatten one island with LSCM.
    public static func lscm(_ mesh: Mesh3, island: UVIsland, options: UVUnwrapOptions = .init()) -> UVIslandLayout {
        let n = island.vertexCount
        var pos = [Vec3](repeating: .zero, count: n)
        var rowsV: [(Int, Int, Int)] = [], wr: [SIMD3<Double>] = [], wi: [SIMD3<Double>] = []
        var area3D = 0.0, nsum = Vec3.zero
        var edgeUse: [UVEdge: Int] = [:]
        for (k, t) in island.triangles.enumerated() {
            let l = (0..<3).map { island.corners[3 * k + $0] }
            let p = (0..<3).map { mesh.positions[Int(mesh.indices[3 * t + $0])] }
            for c in 0..<3 { pos[l[c]] = p[c] }
            for c in 0..<3 where l[c] != l[(c + 1) % 3] { edgeUse[UVEdge(l[c], l[(c + 1) % 3]), default: 0] += 1 }
            let cr = cross3(p[1] - p[0], p[2] - p[0])
            let area = len3(cr) / 2
            guard area > 1e-14 else { continue }
            area3D += area; nsum += cr / 2
            let e1 = p[1] - p[0], l1 = len3(e1), xa = e1 / l1
            let ya = normalize3(cross3(normalize3(cr), xa))
            let q = [Vec2(0, 0), Vec2(l1, 0), Vec2(dot3(p[2] - p[0], xa), dot3(p[2] - p[0], ya))]
            let sq = (l1 * q[2].y).squareRoot()
            let w = [(q[2] - q[1]) / sq, (q[0] - q[2]) / sq, (q[1] - q[0]) / sq]
            rowsV.append((l[0], l[1], l[2]))
            wr.append(SIMD3(w[0].x, w[1].x, w[2].x)); wi.append(SIMD3(w[0].y, w[1].y, w[2].y))
        }
        guard !rowsV.isEmpty else { return UVIslandLayout(island: island, uv: [Vec2](repeating: .zero, count: n)) }

        // Pins: the farthest pair among boundary vertices (all vertices on a closed island).
        var used = Set<Int>(); for r in rowsV { used.insert(r.0); used.insert(r.1); used.insert(r.2) }
        var cand = Set<Int>()
        for (e, c) in edgeUse where c == 1 { cand.insert(e.a); cand.insert(e.b) }
        let pool = (cand.isEmpty ? used : cand.intersection(used)).sorted()
        func farthest(_ from: Int) -> Int { pool.max { len3(pos[$0] - pos[from]) < len3(pos[$1] - pos[from]) }! }
        let pa = farthest(farthest(pool[0])), pb = farthest(pa)
        let dpin = max(len3(pos[pa] - pos[pb]), 1e-12)

        // Start from the planar projection along the mean normal, x toward pb.
        let nAvg = len3(nsum) > 1e-14 ? normalize3(nsum) : Vec3(0, 0, 1)
        var xa = pos[pb] - pos[pa]; xa -= nAvg * dot3(xa, nAvg)
        xa = len3(xa) > 1e-12 ? normalize3(xa) : normalize3(cross3(nAvg, abs(nAvg.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0)))
        let ya = cross3(nAvg, xa)
        var x = [Double](repeating: 0, count: 2 * n)   // u block then v block
        for i in 0..<n { x[i] = dot3(pos[i] - pos[pa], xa); x[n + i] = dot3(pos[i] - pos[pa], ya) }
        var fixed = [Bool](repeating: false, count: 2 * n)
        func pin(_ i: Int, _ u: Double, _ v: Double) { x[i] = u; x[n + i] = v; fixed[i] = true; fixed[n + i] = true }
        pin(pa, 0, 0); pin(pb, dpin, 0)
        for i in 0..<n where !used.contains(i) { fixed[i] = true; fixed[n + i] = true; x[i] = 0; x[n + i] = 0 }

        // y = MᵀM·v (all variables), M's rows being the two real equations per triangle.
        func normalApply(_ v: [Double], into y: inout [Double]) {
            for i in 0..<y.count { y[i] = 0 }
            for (t, r) in rowsV.enumerated() {
                let idx = [r.0, r.1, r.2]
                var a = 0.0, b = 0.0
                for j in 0..<3 {
                    a += wr[t][j] * v[idx[j]] - wi[t][j] * v[n + idx[j]]
                    b += wi[t][j] * v[idx[j]] + wr[t][j] * v[n + idx[j]]
                }
                for j in 0..<3 {
                    y[idx[j]] += wr[t][j] * a + wi[t][j] * b
                    y[n + idx[j]] += -wi[t][j] * a + wr[t][j] * b
                }
            }
        }
        var diag = [Double](repeating: 0, count: 2 * n)
        for (t, r) in rowsV.enumerated() {
            for (j, i) in [r.0, r.1, r.2].enumerated() {
                let d = wr[t][j] * wr[t][j] + wi[t][j] * wi[t][j]
                diag[i] += d; diag[n + i] += d
            }
        }
        var ax = [Double](repeating: 0, count: 2 * n)
        normalApply(x, into: &ax)
        var res = (0..<(2 * n)).map { fixed[$0] ? 0 : -ax[$0] }
        var z = (0..<(2 * n)).map { fixed[$0] || diag[$0] < 1e-300 ? 0 : res[$0] / diag[$0] }
        var p = z
        var rz = zip(res, z).reduce(0) { $0 + $1.0 * $1.1 }
        let r0 = max(res.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-300)
        let maxIt = options.maxIterations ?? min(20_000, 50 * n + 200)
        var ap = [Double](repeating: 0, count: 2 * n)
        var it = 0
        while it < maxIt, res.reduce(0, { $0 + $1 * $1 }).squareRoot() > options.tolerance * r0 {
            normalApply(p, into: &ap)
            for i in 0..<(2 * n) where fixed[i] { ap[i] = 0 }
            let pap = zip(p, ap).reduce(0) { $0 + $1.0 * $1.1 }
            guard pap > 1e-300 else { break }
            let alpha = rz / pap
            for i in 0..<(2 * n) { x[i] += alpha * p[i]; res[i] -= alpha * ap[i] }
            for i in 0..<(2 * n) { z[i] = fixed[i] || diag[i] < 1e-300 ? 0 : res[i] / diag[i] }
            let rzNew = zip(res, z).reduce(0) { $0 + $1.0 * $1.1 }
            let beta = rzNew / rz; rz = rzNew
            for i in 0..<(2 * n) { p[i] = z[i] + beta * p[i] }
            it += 1
        }

        var uv = (0..<n).map { Vec2(x[$0], x[n + $0]) }
        var layout = UVIslandLayout(island: island, uv: uv)
        // Keep winding, then match UV area to surface area.
        var a = layout.uvArea(mesh)
        if a < 0 { uv = uv.map { Vec2($0.x, -$0.y) }; layout.uv = uv; a = -a }
        if a > 1e-300 { let s = (area3D / a).squareRoot(); layout = layout.mapped { $0 * s } }
        if options.alignToAxes { layout = UVGeometry.alignedToMinimumRectangle(layout) }
        return UVUnwrapResult.normalizedToOrigin(layout)
    }
}

/// Everything an unwrap produces.
public struct UVUnwrapResult: Sendable {
    public var mesh: Mesh3
    public var layouts: [UVIslandLayout]
    public var report: UVReport

    static func normalizedToOrigin(_ l: UVIslandLayout) -> UVIslandLayout {
        let lo = l.bounds.min
        return l.mapped { $0 - lo }
    }
}

public extension Mesh3 {
    /// Unwrap every island left by `seams` with LSCM, optionally pack them into 0…1, and write the UVs
    /// (vertices duplicated along seams). `pack: nil` leaves each island at the origin in metres.
    func unwrappedUVs(seams: UVSeams, options: UVUnwrapOptions = .init(),
                      pack: UVPackOptions? = UVPackOptions()) -> UVUnwrapResult {
        let topo = UVTopology(self)
        var layouts = topo.islands(seams).map { UVUnwrap.lscm(self, island: $0, options: options) }
        if let pack { layouts = UVPacker.pack(layouts, mesh: self, options: pack) }
        let out = applying(layouts)
        return UVUnwrapResult(mesh: out, layouts: layouts, report: UVReport(out))
    }

    /// Pack the mesh's existing UV islands (e.g. after a projection) into 0…1.
    func packedUVs(_ options: UVPackOptions = UVPackOptions()) -> UVUnwrapResult {
        let layouts = UVPacker.pack(uvLayouts(), mesh: self, options: options)
        let out = applying(layouts)
        return UVUnwrapResult(mesh: out, layouts: layouts, report: UVReport(out))
    }
}
