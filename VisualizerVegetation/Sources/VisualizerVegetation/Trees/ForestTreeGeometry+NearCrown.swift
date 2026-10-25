import Foundation
import simd

// DH-0629 (Daydream) — how a crown seen from 2–10 m (`TreeSite.nearCamera`) is filled.
//
// Measured on the yard bake (Daydream `ForestTreeRealismTests.testCanopyCoverageReport`, an
// orthographic z-buffer at 1.5 cm a pixel): the hero oak's canopy disc was 20 % SKY although an
// average covered pixel sat behind 8.6 leaf layers — the leaves were there, piled in the wrong
// places. Two constructs did it:
//  · the clump contraction shrank every Voronoi cell of the fill to 46–66 % of its size, so the
//    leaf mass collapsed into separate puffs with sky between them (and the silhouette shrank
//    with them: "broccoli");
//  · the rim seal turned every shell card BROADSIDE-OUT and enlarged it up to 1.55×. Facing the
//    camera that is a regular wall of oversized flat leaves (the "scales" on the front of the
//    crown); at the silhouette — where the seal was meant to close the profile — a card facing
//    outward is EDGE-ON to the viewer, so it closes nothing and draws the streaks along the
//    crown's edge. The seal exists for 25–46 m Forest crowns; a near crown is closed by its spray.
extension ForestTreeGeometry {
    /// Near-crown clump contraction: each fill cluster moves to `c + (p − c)·k`, k ∈ [lo, lo+span].
    /// 1 = no clumping. Was 0.46…0.66.
    public nonisolated(unsafe) static var clumpContraction: (lo: Float, span: Float) = (0.92, 0.08)
    /// Near crowns keep the hemispherical spray on their shell cards (no broadside-out turn, ≤ 1.12×
    /// enlargement). false = the Forest rim seal on near crowns too (the pre-DH-0629-fill look).
    public nonisolated(unsafe) static var nearCrownSprayRim = true
    /// Each clump is a rounded MASS of radius `clumpRadiusFrac` × the distance to its nearest
    /// neighbour clump: clusters beyond it are moved back inside it (`clumpOffset`). Neighbouring
    /// masses then overlap where clumps are close and leave sky only where three of them fail to
    /// meet — gaps BETWEEN masses, a leaf-mass outline at the silhouette instead of a halo of
    /// stray clusters. 0 disables the clamp.
    public nonisolated(unsafe) static var clumpRadiusFrac: Float = 0.62
    /// A near BIRCH's fill: its crown-area scaling (`crownRBase` × 0.62) dropped it to the 2 000
    /// cluster floor, and the yard birch read 33 % sky — a bare lattice, not a light, lacy crown
    /// (real birches let light through in flecks, not in sheets).
    public nonisolated(unsafe) static var nearBirchFillBoost: Float = 1.8
    /// Near-crown whorl: each blade's centre sits this many blade-lengths out along its petiole.
    public nonisolated(unsafe) static var nearWhorlReach: Float = 0.45

    /// Radius per clump from its nearest-neighbour distance (O(K²), K ≤ 60).
    static func sizeClumps(_ clumps: inout [(c: SIMD3<Float>, k: Float, r: Float)]) {
        guard clumpRadiusFrac > 0, clumps.count > 1 else { return }
        for i in clumps.indices {
            var nn = Float.greatestFiniteMagnitude
            for j in clumps.indices where j != i { nn = min(nn, simd_distance(clumps[i].c, clumps[j].c)) }
            clumps[i].r = nn * clumpRadiusFrac
        }
    }

    /// A cluster's offset from its clump centre: contracted by `k`; a cluster still outside the
    /// clump's mass (`radius`, 0 = no mass) is moved INTO it along its own direction, at a radius
    /// drawn uniformly by volume from [0.35, 1]·radius (`u` ∈ [0, 1), the cluster's own hash).
    /// Clamping onto the surface instead piled the strays into a dense skin — each clump read as a
    /// hollow green ball with a bright rim; resampled into the volume, the mass stays leafy through.
    @inline(__always)
    static func clumpOffset(_ off: SIMD3<Float>, k: Float, radius: Float, u: Float) -> SIMD3<Float> {
        let o = off * k
        guard radius > 0 else { return o }
        let d = simd_length(o)
        guard d > radius else { return o }
        let lo: Float = 0.35 * 0.35 * 0.35
        let d2 = radius * cbrt(lo + (1 - lo) * u)
        return o * (d2 / d)
    }
}
