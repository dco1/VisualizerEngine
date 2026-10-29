import Foundation

/// A cloth's MATERIAL — how stiff it is, how it bends, how it loses energy and whether a fold takes a set — as ONE value,
/// the way `RestPolicy` is one value for when a sim may stop. `solver.apply(.terry)` sets every dial it names; a host
/// that wants one dial different copies the preset and changes that field.
///
/// Rates are per simulated SECOND, so a fabric means the same thing whatever substep a host runs.
public struct ClothFabric: Equatable, Sendable {
    // ── Stiffness: XPBD compliance (0 = rigid, larger = softer) ──
    /// Along the weave. Woven cloth barely stretches: keep it small.
    public var stretchCompliance: Float
    /// Cell diagonals: how freely the weave skews.
    public var shearCompliance: Float
    /// Skip-one links: resistance to bending at the scale of two cells.
    public var bendCompliance: Float
    /// Dihedral hinges: resistance to bending at each edge — what makes a thick fabric round its folds.
    public var dihedralCompliance: Float

    // ── Energy loss ──
    /// Share of a link's relative velocity kept after one second (1 = none, 0 = dead). See `PaperClothSolver.internalDamping`.
    public var internalDamping: Float
    /// Share of ALL velocity kept after one second — air drag. Slows the fall too; keep it near 1 and damp the cloth
    /// with `internalDamping` instead.
    public var airDamping: Float

    // ── Set ──
    /// A hinge bent further than this from its rest angle (radians) starts to take a set. Small = folds stay where they land.
    public var creaseYield: Float
    /// Share of the excess past `creaseYield` absorbed into the rest angle per constraint pass.
    public var creaseCreep: Float

    public init(stretchCompliance: Float, shearCompliance: Float, bendCompliance: Float, dihedralCompliance: Float,
                internalDamping: Float, airDamping: Float, creaseYield: Float, creaseCreep: Float) {
        self.stretchCompliance = stretchCompliance; self.shearCompliance = shearCompliance
        self.bendCompliance = bendCompliance; self.dihedralCompliance = dihedralCompliance
        self.internalDamping = internalDamping; self.airDamping = airDamping
        self.creaseYield = creaseYield; self.creaseCreep = creaseCreep
    }
}

extension ClothFabric {
    /// Terry towelling: heavy, dead, and its folds stay folds. Calibrated on Daydream Home's towel-stack drop (three
    /// folded 40 × 70 cm towels from 0.85 m, 128 substeps a frame) and chosen by eye over the flat-rest arm, which
    /// dropped like rubber. What makes it terry rather than rubber is the last three dials: bending energy dies in the cloth
    /// (internalDamping 1e-4 — the hinge damper), and a fold bent past 0.05 rad takes a set where it lands. Pair it with
    /// `restBendingToCurrentShape()` when the cloth starts pre-folded.
    public static let terry = ClothFabric(stretchCompliance: 1e-6, shearCompliance: 8e-6,
                                          bendCompliance: 4e-4, dihedralCompliance: 0.15,
                                          internalDamping: 1e-4, airDamping: 0.548,
                                          creaseYield: 0.05, creaseCreep: 0.4)
}

extension PaperClothSolver {
    /// Make the cloth this fabric: its stiffness (rewritten into links already built, and into any `configureSheets`
    /// builds later), its damping and its set. `damping` is derived from `fixedDt`, so set `fixedDt` first.
    public func apply(_ f: ClothFabric) {
        fabric = f
        damping = pow(min(max(f.airDamping, 0), 1), fixedDt)
        internalDamping = f.internalDamping
        dihedralYieldAngle = f.creaseYield
        dihedralCreepRate = f.creaseCreep
        applyStiffness(f)
    }

    /// Rewrite the compliance of every cloth link already built (colour groups 0…13: stretch, shear, skip-one bend —
    /// see `configureSheets`) and every hinge. Struts (stuffing) keep theirs.
    func applyStiffness(_ f: ClothFabric) {
        let c = constraintBuffer.contents
        for g in 0 ..< min(14, groupCount.count) {
            let alpha = g < 4 ? f.stretchCompliance : g < 8 ? f.shearCompliance : f.bendCompliance
            for k in groupStart[g] ..< groupStart[g] + groupCount[g] { c[k].compliance = alpha }
        }
        let b = bendConstraintBuffer.contents
        for k in 0 ..< bendConstraintBuffer.count { b[k].compliance = f.dihedralCompliance }
    }
}
