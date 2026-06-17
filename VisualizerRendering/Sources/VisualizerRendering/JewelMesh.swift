import Foundation
import simd

// ── JewelMesh ─────────────────────────────────────────────────────────────────
//
// Unit gemstone geometry for the instanced renderer — a faceted ROUND BRILLIANT
// CUT diamond whose axis is local +Y (identity-oriented gems sit table-up). Built
// to the SAME `CoinMesh.Mesh` contract (positions/normals/uvs/indices) so it drops
// straight into `CoinInstancedRenderer` + `CoinDEMSolver` with no GPU changes: the
// Coin Pusher's jewel drop-asset is a second solver/renderer pair fed this mesh.
//
// FLAT facet normals (one per facet, vertices NOT shared between facets) are the
// whole point: a brilliant's brilliance is dozens of mirror-flat faces each
// catching a different sharp glint as the stone tumbles under the arcade neon. A
// smooth-shaded gem reads as a blob of plastic; a flat-faceted one sparkles.
//
// The cut follows real round-brilliant proportions (relative to the girdle
// radius): table ~57%, crown ~16% of diameter, pavilion ~43% of diameter, a thin
// girdle band, converging to a point culet. Geometry is CENTERED on y=0 so the
// solver's symmetric oriented-cylinder collider (radius = girdle R, halfThickness
// = half the stone's height) wraps it — see CoinPusherController's jewel stream.
//
// Authored DOUBLE-SIDED (each facet emitted with both windings) for the same
// reason the cabinet quads are: the gem is a closed convex solid, so back faces
// are depth-occluded by front faces and never seen — emitting both windings makes
// the mesh winding-agnostic and sidesteps the inside-out-mesh invisibility trap
// (CLAUDE.md gap #4 / scn_geometry_winding) entirely. Normals always point
// geometrically OUTWARD, so the visible (front) facets shade correctly.

public enum JewelMesh {

    /// Build a unit round-brilliant-cut diamond.
    /// - `girdleRadius`: the stone's widest radius (the girdle); the collider's
    ///   `coinRadius`. Everything else scales from real cut proportions.
    /// - `segments`: n-fold symmetry of the crown (8 = classic). The girdle and
    ///   pavilion use `2·segments` facets for finer sparkle.
    public static func brilliant(girdleRadius R: Float = 0.14,
                                 tableRatio: Float = 0.60,
                                 crownRatio: Float = 0.33,
                                 girdleRatio: Float = 0.07,
                                 pavilionRatio: Float = 0.88,
                                 segments n: Int = 8) -> CoinMesh.Mesh {
        let nMain = max(4, n)
        let nGird = nMain * 2
        let tableR   = tableRatio * R
        let crownH   = crownRatio * R
        let girdleH  = girdleRatio * R
        let pavilionH = pavilionRatio * R

        // Pre-centre y levels so the bounding box is symmetric about y = 0.
        let yTable0  =  girdleH * 0.5 + crownH
        let yCG0     =  girdleH * 0.5            // crown side of the girdle band
        let yPG0     = -girdleH * 0.5            // pavilion side of the girdle band
        let yCulet0  = -girdleH * 0.5 - pavilionH
        let yMid     = (yTable0 + yCulet0) * 0.5
        let yTable  = yTable0  - yMid
        let yCG     = yCG0     - yMid
        let yPG     = yPG0     - yMid
        let yCulet  = yCulet0  - yMid

        func ring(_ count: Int, radius: Float, y: Float, phase: Float = 0) -> [SIMD3<Float>] {
            (0..<count).map { k in
                let a = phase + Float(k) / Float(count) * 2 * .pi
                return SIMD3(radius * cos(a), y, radius * sin(a))
            }
        }

        // Table octagon corners aligned to the even girdle indices (every main
        // azimuth) so the crown bezel facets line up under the table corners.
        let table = ring(nMain, radius: tableR, y: yTable)
        let cg    = ring(nGird, radius: R,      y: yCG)     // crown-side girdle ring
        let pg    = ring(nGird, radius: R,      y: yPG)     // pavilion-side girdle ring
        let culet = SIMD3<Float>(0, yCulet, 0)
        let halfHeight = yTable                              // == -yCulet (centred)

        var pos: [SIMD3<Float>] = []
        var nrm: [SIMD3<Float>] = []
        var uv:  [SIMD2<Float>] = []
        var idx: [Int32] = []

        func uvOf(_ p: SIMD3<Float>) -> SIMD2<Float> {
            let a = atan2(p.z, p.x) / (2 * .pi) + 0.5
            let v = (p.y / (2 * halfHeight)) + 0.5
            return SIMD2(a, v)
        }

        // Emit one flat facet (3+ coplanar verts, fan-triangulated, DOUBLE-SIDED).
        // The flat normal is forced geometrically OUTWARD: for the radial faces
        // (crown / girdle / pavilion) "outward" is away from the Y axis; the table
        // is horizontal, so it takes an explicit +Y normal.
        func facet(_ verts: [SIMD3<Float>], forcedNormal: SIMD3<Float>? = nil) {
            var nf: SIMD3<Float>
            if let fn = forcedNormal {
                nf = fn
            } else {
                nf = simd_normalize(simd_cross(verts[1] - verts[0], verts[2] - verts[0]))
                let centroid = verts.reduce(SIMD3<Float>(0, 0, 0), +) / Float(verts.count)
                let radial = SIMD3<Float>(centroid.x, 0, centroid.z)
                if simd_length(radial) > 1e-5,
                   simd_dot(nf, simd_normalize(radial)) < 0 { nf = -nf }
            }
            let base = Int32(pos.count)
            for p in verts { pos.append(p); nrm.append(nf); uv.append(uvOf(p)) }
            for k in 1..<(verts.count - 1) {
                let a = base, b = base + Int32(k), c = base + Int32(k + 1)
                idx += [a, b, c,  a, c, b]   // double-sided
            }
        }

        // ── Table (top octagon) ───────────────────────────────────────────────
        facet(table, forcedNormal: SIMD3(0, 1, 0))

        // ── Crown facets (table edge → girdle scallops) ───────────────────────
        // Each main sector tiles to three faceted triangles spanning the two table
        // corners and the three girdle points between them — a star/bezel read.
        for i in 0..<nMain {
            let t0 = table[i], t1 = table[(i + 1) % nMain]
            let g0 = cg[(2 * i) % nGird]
            let g1 = cg[(2 * i + 1) % nGird]
            let g2 = cg[(2 * i + 2) % nGird]
            facet([t0, g0, g1])
            facet([t0, g1, t1])
            facet([t1, g1, g2])
        }

        // ── Girdle band (thin vertical facets) ────────────────────────────────
        for j in 0..<nGird {
            let jn = (j + 1) % nGird
            facet([pg[j], pg[jn], cg[jn], cg[j]])
        }

        // ── Pavilion facets (girdle → culet point) ────────────────────────────
        for j in 0..<nGird {
            let jn = (j + 1) % nGird
            facet([culet, pg[j], pg[jn]])
        }

        return CoinMesh.Mesh(positions: pos, normals: nrm, uvs: uv, indices: idx)
    }

    /// Half the overall vertical extent of a `brilliant` stone at the given girdle
    /// radius + proportions — the value to hand the solver as `halfThickness` so
    /// its oriented-cylinder collider wraps the stone (table-up rest, culet down).
    public static func halfHeight(girdleRadius R: Float = 0.14,
                                  crownRatio: Float = 0.33,
                                  girdleRatio: Float = 0.07,
                                  pavilionRatio: Float = 0.88) -> Float {
        (crownRatio + girdleRatio + pavilionRatio) * R * 0.5
    }
}
