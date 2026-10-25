import Foundation

/// **Where a texel sits in a laid-unit pattern** — the one piece of grid arithmetic every
/// laid-unit bake needs, written once.
///
/// Tile and brick are the same object (units of a real size, laid in a bond, separated by a joint;
/// see `UnitPatternParams`), and they differ only in what the unit's FACE is: a glaze or a fired
/// clay. So the classification of a texel — which unit it belongs to, where inside that unit it
/// is, how far it is from the nearest joint, and how much of it IS joint — lives here, and each
/// generator supplies only its own face. Before this, the course-offset and lattice arithmetic
/// lived inline in `MaterialGenerator.tile` / `tileLattice`, so a second laid-unit material had a
/// choice between re-deriving it (a second grid generator, free to drift — the paver bake already
/// had: it ignores bond and run entirely) or not existing. `tile` and `brick` both sample through
/// this, and `tile` is byte-identical to its pre-extraction bake (DH-1497).
///
/// Every distance is kept in the units the tile bake has always used, so nothing about tile moves:
///  - **course bonds** (stack / running / ⅓): `dU` in LONG-edge fractions, `dV` in SHORT-edge
///    fractions (multiply `dV` by `aspect` to compare it with `dU` physically);
///  - **lattices** (herringbone / diagonal): one `dMin` in the lattice's own cell units.
/// `rimMeters` is the same distance in metres for a face that reasons physically (brick's arris).
public struct UnitLattice: Sendable {
    public let pattern: UnitPatternParams
    public let layout: TileLayout
    public let size: Int

    // Course-bond joint geometry, in tile units (see `MaterialGenerator.tile` for why the joint is
    // floored at `TileLayout.minJointTexels` and why the v half-width scales by the aspect).
    public let aspect: Double
    public let halfU: Double, halfV: Double
    public let softU: Double, softV: Double
    // Lattice joint geometry, in lattice cell units.
    public let latticeCells: Int
    public let latticeHalf: Double
    public let latticeSoft: Double

    public init(pattern p: UnitPatternParams, layout: TileLayout, size: Int) {
        self.pattern = p; self.layout = layout; self.size = size
        aspect = Double(p.shape.aspect)
        let texelsU = Double(size) / Double(Swift.max(1, layout.cols))
        let texelsV = Double(size) / Double(Swift.max(1, layout.rows))
        let jf = layout.effectiveJointFraction(p.jointFraction)
        halfU = jf / 2
        halfV = jf * aspect / 2
        // Antialias band, in tile units, ≈1 texel wide on each axis.
        softU = Swift.max(1.0 / Swift.max(texelsU, 1), 0.004)
        softV = Swift.max(1.0 / Swift.max(texelsV, 1), 0.004)

        let herring = p.bond == .herringbone
        let cells = Swift.max(1, layout.cols)
        latticeCells = cells
        let texelsPerCell = Double(size) / Double(cells)
        latticeSoft = Swift.max((herring ? 1.0 : 1.4142) / Swift.max(texelsPerCell, 1), 0.004)
        // Herringbone: joint is `jf` of the LONG (L-cell) edge, measured in cells. Diagonal: `jf` of
        // the diamond edge, in `a`/`b` units, where one cell of `a` spans edge/√2 → ×√2. NOT the
        // floored `effectiveJointFraction` — the lattice floors in its own cell units below.
        let L = TileLayout.herringboneBrickLength
        let half = herring ? p.jointFraction * Double(L) / 2 : p.jointFraction * 1.4142 / 2
        latticeHalf = Swift.max(half, TileLayout.minJointTexels / (2 * Swift.max(texelsPerCell, 1)))
    }

    /// The sampling coordinate of texel `(x, y)`: texel-centre (so column 0 and column `size-1`
    /// land symmetrically about the seam), and swapped for a VERTICAL run — an exact quarter turn
    /// of the lattice, which on a square bake maps the pattern onto itself.
    public func uv(x: Int, y: Int) -> (u: Double, v: Double) {
        let su = (Double(x) + 0.5) / Double(size)
        let sv = (Double(y) + 0.5) / Double(size)
        return pattern.run == .vertical ? (sv, su) : (su, sv)
    }

    /// One texel's place in the pattern.
    public struct Cell: Sendable {
        /// Within-unit coordinate, 0…1 along the unit's long axis (`fx`) and short axis (`fy`).
        public var fx: Double, fy: Double
        /// Wrapped per-unit identity — identical either side of the bake seam, so a per-unit hash
        /// of it is seamless.
        public var keyI: Int, keyJ: Int
        /// Course bonds: distance to the nearest joint centre on each axis (see the type note).
        /// Lattices: both equal `dMin`.
        public var dU: Double, dV: Double
        /// Joint coverage, 0 (unit face) … 1 (joint), antialiased over ≈1 texel.
        public var joint: Double
        /// Distance to the joint relative to its half-width — 1 at the joint's edge, growing into
        /// the face. What a glaze pools against.
        public var edgeN: Double
        /// The tile bevel's and handmade cushion's rim coordinate: `min(dU, dV × aspect)` on the
        /// course bonds, `dMin` on the lattices. Kept EXACTLY as tile has always computed it so the
        /// tile bake does not move — note it is not a physical distance on a rectangular unit
        /// (`dV × aspect` over-weights the short axis; filed against tile's bevel). Physical faces
        /// read `rimMeters`.
        public var rim: Double
        /// The same distance in METRES, measured to the joint CENTRE; subtract `jointHalfMeters`
        /// for the distance to the unit's arris.
        public var rimMeters: Double
        /// True on the lattice arrangements (herringbone / diagonal).
        public var isLattice: Bool
    }

    /// Half the realised joint width in metres — the same on both axes by construction.
    public var jointHalfMeters: Double {
        pattern.bond.isLattice
            ? latticeHalfMeters
            : halfU * layout.realisedSizeMeters
    }

    private var latticeHalfMeters: Double {
        // One lattice cell spans `run / cells` metres on the bake axis; a diagonal's a/b unit is
        // that over √2.
        let cellM = layout.runMeters / Double(latticeCells)
        return pattern.bond == .herringbone ? latticeHalf * cellM : latticeHalf * cellM / 1.4142
    }

    public func cell(u: Double, v: Double) -> Cell {
        let p = pattern
        if p.bond.isLattice { return latticeCell(u: u, v: v) }
        let cols = layout.cols, rows = layout.rows
        let tv = v * Double(rows)
        let tj = Int(floor(tv))
        let tu = u * Double(cols) + p.bond.offset(row: tj)
        let ti = Int(floor(tu))
        let fx = tu - floor(tu), fy = tv - floor(tv)
        let tiW = ((ti % cols) + cols) % cols
        let tjW = ((tj % rows) + rows) % rows
        let dU = Swift.min(fx, 1 - fx), dV = Swift.min(fy, 1 - fy)
        let gU = 1 - smoothstep(halfU, halfU + softU, dU)
        let gV = 1 - smoothstep(halfV, halfV + softV, dV)
        let edgeN = Swift.min(dU / Swift.max(halfU, 1e-6), dV / Swift.max(halfV, 1e-6))
        let rim = Swift.min(dU, dV * aspect)
        // Physical distance: dU is a fraction of the LONG edge, dV of the SHORT edge.
        let longM = layout.realisedSizeMeters
        let rimM = Swift.min(dU * longM, dV * longM / aspect)
        return Cell(fx: fx, fy: fy, keyI: tiW, keyJ: tjW, dU: dU, dV: dV,
                    joint: Swift.max(gU, gV), edgeN: edgeN, rim: rim,
                    rimMeters: rimM, isLattice: false)
    }

    /// Herringbone and the 45° diagonal — their own lattices, not course offsets. See
    /// `MaterialGenerator.tileLattice` for the wrap arithmetic.
    private func latticeCell(u: Double, v: Double) -> Cell {
        let cells = latticeCells
        var dMin = 0.0, faceU = 0.0, faceV = 0.0, keyI = 0, keyJ = 0
        if pattern.bond == .herringbone {
            let L = TileLayout.herringboneBrickLength
            let gx = u * Double(cells), gy = v * Double(cells)
            let i = Int(floor(gx)), j = Int(floor(gy))
            let s = (((i + j) % (2 * L)) + 2 * L) % (2 * L)
            let ax: Int, ay: Int, lu: Int, lv: Int
            if s < L { ax = i - s; ay = j; lu = L; lv = 1 }      // horizontal brick
            else     { ax = i;     ay = j - (s - L); lu = 1; lv = L }  // vertical brick
            let lx = gx - Double(ax), ly = gy - Double(ay)
            let dU = Swift.min(lx, Double(lu) - lx)
            let dV = Swift.min(ly, Double(lv) - ly)
            dMin = Swift.min(dU, dV)
            faceU = lx / Double(lu); faceV = ly / Double(lv)
            keyI = ((ax % cells) + cells) % cells
            // Fold orientation in so an H brick and a V brick sharing an anchor coordinate
            // still draw as two different cuts.
            keyJ = (((ay % cells) + cells) % cells) + (s < L ? 0 : cells)
        } else {
            let a = (u + v) * Double(cells)
            let b = (u - v) * Double(cells)
            let fa = a - floor(a), fb = b - floor(b)
            dMin = Swift.min(Swift.min(fa, 1 - fa), Swift.min(fb, 1 - fb))
            faceU = fa; faceV = fb
            keyI = ((Int(floor(a)) % cells) + cells) % cells
            keyJ = ((Int(floor(b)) % cells) + cells) % cells
        }
        let joint = 1 - smoothstep(latticeHalf, latticeHalf + latticeSoft, dMin)
        let cellM = layout.runMeters / Double(cells)
        let rimM = pattern.bond == .herringbone ? dMin * cellM : dMin * cellM / 1.4142
        return Cell(fx: faceU, fy: faceV, keyI: keyI, keyJ: keyJ, dU: dMin, dV: dMin,
                    joint: joint, edgeN: dMin / Swift.max(latticeHalf, 1e-6), rim: dMin,
                    rimMeters: rimM, isLattice: true)
    }
}
