import Foundation

/// The reed profile of FLUTED (reeded) architectural glass — the ONE description of the
/// flute shape in the codebase.
///
/// **Why this exists.** Obscuring glass used to be faked as a *rough dielectric*
/// (`roughness 0.40`). A rough dielectric is estimated STOCHASTICALLY: the RT-glass pass
/// jitters each refraction/reflection ray in a cone whose width grows with roughness², one
/// (or a few) samples per pixel per frame, denoised only by TAA. That is precisely what
/// Danny saw as "live noise" — the pane fizzes, and it fizzes *differently* every frame
/// because the estimator is re-seeded per frame (`u.frameSeed`).
///
/// Real reeded glass isn't rough at all. It is optically SMOOTH glass whose room face is
/// moulded into vertical half-round ridges: the pattern is *shape*, not *scatter*. So we
/// model it as shape. `HouseMesh.glazing` extrudes a scalloped plan footprint derived from
/// this profile — every plan lobe becomes a full-height flute — and the glass stays at the
/// clear pane's optical roughness. The result is deterministic per frame: identical camera
/// ⇒ identical pixels (the numeric definition of "no live noise"), and the RT refraction
/// bends through the real ridge normals, so the flutes distort the world behind them the
/// way reeded glass actually does.
///
/// `MaterialGenerator.flutedGlass` bakes its albedo/normal/roughness from the SAME profile,
/// so the picker swatch and the pane in the render can never drift apart.
public enum FlutedGlassProfile {

    /// Reed pitch — centre-to-centre spacing of the vertical flutes, in metres.
    /// 30 mm is the common architectural "reeded" spacing (narrower reads as ribbed).
    public static let pitch: Double = 0.030

    /// Peak relief of one reed above the pane's flat back face, in metres.
    /// ~3.5 mm gives a visible half-round without making the pane read as corrugated.
    public static let depth: Double = 0.0035

    /// Polyline segments per reed when the profile is realised as geometry. EVEN, so one
    /// sample lands exactly on each reed crest (an odd count straddles it and flattens every
    /// lobe); 6 keeps the chord error of a 3.5 mm lobe under half a millimetre while the
    /// pane's triangle count stays in the low thousands.
    public static let segmentsPerReed = 6

    /// Reed height above the flat plane at distance `s` (metres) along the pane.
    /// One smooth half-round lobe per `pitch`, meeting zero at every seam — so
    /// consecutive reeds join without a crack and the pane stays watertight.
    @inlinable public static func relief(at s: Double) -> Double {
        let phase = s / pitch
        return depth * sin(Double.pi * (phase - floor(phase)))
    }

    /// d(relief)/ds at `s` — the surface slope the material bake turns into a
    /// tangent-space normal. Same profile, differentiated; never re-authored.
    @inlinable public static func slope(at s: Double) -> Double {
        let phase = s / pitch
        return depth * Double.pi / pitch * cos(Double.pi * (phase - floor(phase)))
    }

    /// Number of polyline samples across a pane of `length` metres (≥ 2). The reed count is
    /// ROUNDED to a whole number, and the caller spaces samples evenly over the real length —
    /// so the pitch stretches by at most half a reed to fit the opening and no window ever
    /// ends on a sliced-off partial flute (which is also how reeded glass is really cut).
    @inlinable public static func sampleCount(forLength length: Double) -> Int {
        let reeds = max(1.0, (length / pitch).rounded())
        return max(2, Int(reeds) * segmentsPerReed + 1)
    }
}
