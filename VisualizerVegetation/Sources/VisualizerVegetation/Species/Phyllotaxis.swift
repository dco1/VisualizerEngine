import Foundation

/// How a plant spaces successive organs around an axis — the arrangement layer's shared arithmetic.
///
/// This exists because the golden angle was written down **six times in three different values**,
/// and two of those disagreements were inside a single subsystem:
///
///     Double.pi * (3 - sqrt(5))   2.3999632297…   Mesh3+Ray, PottedPlantMesh ×2
///     2.399963                    (Float)         ForestTreeGeometry+Crown ×2
///     2.39996                     (Float)         ForestTreeGeometry+Foliage
///
/// So the yard tree's crown fill and its twig phyllotaxis spiralled at angles differing by
/// 3.2 × 10⁻⁶ rad — not because anyone chose that, but because the constant was retyped and
/// truncated differently on two occasions. That is the drift this module exists to stop, in its
/// smallest and most literal form: a fact typed twice will differ.
///
/// The value is DERIVED, never a literal. π(3 − √5) is exact to the last bit the type can hold,
/// and a transcribed decimal is not — which is precisely how the three spellings arose.
public enum Phyllotaxis {

    /// The golden angle, π(3 − √5) ≈ 2.399963 rad ≈ 137.5°.
    ///
    /// Successive organs placed this far apart around an axis never fall into radial rows, at any
    /// count — which is why real plants use it and why a leaf spiral built on a rounder number
    /// reads as banded. Also the right constant for a Fibonacci lattice on a sphere.
    public static let goldenAngle: Double = .pi * (3 - 5.0.squareRoot())

    /// The golden angle in single precision, for the `Float` geometry paths.
    ///
    /// `Float(goldenAngle)` rather than a transcribed decimal: rounding the exact value once is
    /// correct by construction, whereas typing 2.39996 introduces an error 20× larger than the
    /// representation itself.
    public static let goldenAngleF: Float = Float(.pi * (3 - 5.0.squareRoot()))

    /// The azimuth of the `index`-th organ on a golden spiral, offset by `start`.
    public static func spiralAzimuth(index: Int, start: Double = 0) -> Double {
        start + Double(index) * goldenAngle
    }

    /// `Float` overload of `spiralAzimuth(index:start:)`.
    public static func spiralAzimuth(index: Int, start: Float) -> Float {
        start + Float(index) * goldenAngleF
    }

    /// Bias a 0…1 station parameter toward the TIP of the axis.
    ///
    /// Leaves are not spaced evenly along a stem or a twig: they crowd toward the growing end,
    /// because that is where the recent growth is. Raising the parameter to a power above 1 does
    /// that — and the exponent is a real per-species difference (the shipped houseplants use 1.35,
    /// the tree's twigs 1.45), so it is a parameter here rather than a constant.
    ///
    /// `exponent == 1` is an exact no-op: even spacing.
    public static func tipBiased(_ t: Double, exponent: Double) -> Double {
        exponent == 1 ? t : pow(t, exponent)
    }

    /// `Float` overload of `tipBiased(_:exponent:)`.
    public static func tipBiased(_ t: Float, exponent: Float) -> Float {
        exponent == 1 ? t : pow(t, exponent)
    }
}
