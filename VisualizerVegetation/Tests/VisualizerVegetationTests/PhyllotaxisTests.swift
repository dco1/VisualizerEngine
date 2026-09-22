import XCTest
import Foundation
@testable import VisualizerVegetation

/// Gates for the shared phyllotaxis arithmetic.
///
/// These exist because the golden angle had been written down six times in three different values,
/// and two of those disagreed inside a single subsystem — the yard tree's crown fill used
/// `2.399963` while its twig phyllotaxis used `2.39996`, so the two halves of one tree spiralled at
/// angles 3.2e-06 rad apart. Nobody chose that; the constant was retyped and truncated differently
/// on two occasions.
final class PhyllotaxisTests: XCTestCase {

    /// The constant is DERIVED, and the `Float` form is the derived value rounded ONCE — never a
    /// transcribed decimal. This is the specific regression: `2.39996` is off by 3.2e-06, which is
    /// about twenty times coarser than `Float`'s own resolution at that magnitude, so a transcribed
    /// literal is materially worse than simply converting the exact value.
    func testGoldenAngleIsDerivedNotTranscribed() {
        XCTAssertEqual(Phyllotaxis.goldenAngle, .pi * (3 - 5.0.squareRoot()), accuracy: 0)
        XCTAssertEqual(Phyllotaxis.goldenAngleF, Float(Phyllotaxis.goldenAngle), accuracy: 0,
                       "goldenAngleF must be Float(goldenAngle) — if this fails, someone typed a "
                       + "decimal instead of converting, which is exactly how the three spellings "
                       + "of this constant arose")

        // The two historical truncations, pinned so their error stays legible.
        XCTAssertEqual(abs(Phyllotaxis.goldenAngle - 2.39996), 3.2297e-06, accuracy: 1e-9)
        XCTAssertGreaterThan(abs(Phyllotaxis.goldenAngle - 2.39996),
                             Double(Float.ulpOfOne) * 2.4,
                             "the old literal's error should exceed Float's own resolution — that "
                             + "is why it was worth fixing rather than tolerating")
    }

    /// **The property that makes this constant the right one:** successive organs on a golden
    /// spiral never fall into radial rows, at ANY count. A spiral that bands reads as a
    /// machine-made plant.
    ///
    /// The mechanism is irrationality, not roundness — the golden angle is the hardest number to
    /// approximate by a fraction, so the stations never land on a finite set of spokes. An angle
    /// that IS a rational fraction of a turn repeats after its denominator and bands immediately.
    ///
    /// Measured as the smallest angular gap between any two of the first N stations, wrapped to a
    /// circle: for the golden angle it stays open at every N; for a rational angle it is zero.
    func testGoldenSpiralNeverFallsIntoRadialRows() {
        func tightestGap(angle: Double, stations: Int) -> Double {
            let twoPi = 2 * Double.pi
            let az = (0 ..< stations).map { fmod(Double($0) * angle, twoPi) }.sorted()
            var tightest = twoPi
            for i in 0 ..< az.count {
                let next = i + 1 < az.count ? az[i + 1] : az[0] + twoPi
                tightest = min(tightest, next - az[i])
            }
            return tightest
        }

        for n in [12, 34, 89, 233] {
            let golden = tightestGap(angle: Phyllotaxis.goldenAngle, stations: n)
            // An even spread of n points would gap 2pi/n; the golden spiral stays within ~2.3x of
            // that ideal at every count, which is the property that keeps it from banding.
            XCTAssertGreaterThan(golden, (2 * .pi / Double(n)) / 2.4,
                                 "golden spiral banded at \(n) stations")
        }

        // The failure mode this constant avoids is RATIONAL approximation. An angle that is a
        // rational fraction of a turn repeats after its denominator: 2/5 of a turn puts every
        // station on one of five spokes, and the plant reads as machine-made.
        //
        // (Note what does NOT work as a comparator: 2.4 rad. It looks like a "rounder" number but
        // it is 3.7e-05 rad from the golden angle — essentially the same angle — and it spreads
        // just as well. The first version of this test asserted the golden angle beat it and was
        // simply wrong. The property is irrationality, not roundness.)
        let fiveSpokes = 2 * Double.pi * 2.0 / 5.0
        XCTAssertEqual(tightestGap(angle: fiveSpokes, stations: 200), 0, accuracy: 1e-9,
                       "a rational angle must collapse onto spokes — otherwise this comparison "
                       + "proves nothing")
        XCTAssertGreaterThan(tightestGap(angle: Phyllotaxis.goldenAngle, stations: 200), 0.019,
                             "the golden angle must keep every station separated at 200 stations")
    }

    func testSpiralAzimuthMatchesTheDefinitionInBothPrecisions() {
        for i in [0, 1, 7, 50] {
            XCTAssertEqual(Phyllotaxis.spiralAzimuth(index: i),
                           Double(i) * Phyllotaxis.goldenAngle, accuracy: 1e-12)
            XCTAssertEqual(Phyllotaxis.spiralAzimuth(index: i, start: Float(0)),
                           Float(i) * Phyllotaxis.goldenAngleF, accuracy: 0)
        }
        XCTAssertEqual(Phyllotaxis.spiralAzimuth(index: 3, start: 1.0),
                       1.0 + 3 * Phyllotaxis.goldenAngle, accuracy: 1e-12)
    }

    /// `exponent == 1` must be an EXACT no-op, not `pow(t, 1)` — an even-spacing caller should get
    /// its input back bit-for-bit rather than whatever `pow` rounds to.
    func testTipBiasIsExactAtExponentOne() {
        for t in [0.0, 0.1, 0.37, 0.5, 0.9, 1.0] {
            XCTAssertEqual(Phyllotaxis.tipBiased(t, exponent: 1), t, accuracy: 0)
            XCTAssertEqual(Phyllotaxis.tipBiased(Float(t), exponent: 1), Float(t), accuracy: 0)
        }
        // And it does bias toward the tip above 1: interior values move DOWN, so stations crowd
        // toward the end of the run.
        for t in [0.2, 0.5, 0.8] {
            XCTAssertLessThan(Phyllotaxis.tipBiased(t, exponent: 1.35), t)
            XCTAssertLessThan(Phyllotaxis.tipBiased(t, exponent: 1.45),
                              Phyllotaxis.tipBiased(t, exponent: 1.35),
                              "a larger exponent must crowd harder")
        }
        // Endpoints are fixed under any exponent.
        XCTAssertEqual(Phyllotaxis.tipBiased(0.0, exponent: 1.45), 0, accuracy: 0)
        XCTAssertEqual(Phyllotaxis.tipBiased(1.0, exponent: 1.45), 1, accuracy: 1e-15)
    }
}
