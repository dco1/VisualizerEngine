import XCTest
import simd
@testable import VisualizerRendering

/// The LTC area-light table is fitted OFFLINE and checked in as literals
/// (`IlluminatoramaLTCTable`) rather than re-derived by a Nelder-Mead optimizer in
/// every `IlluminatoramaRenderer.init` — that fit was 2639 ms of Daydream Home's
/// 2776 ms cold launch at `-Onone`.
///
/// A checked-in derived constant is only safe if something proves it still matches
/// what derives it. That is this file: it runs the REAL fitter and diffs it against
/// the literals. Change the GGX BRDF, the error metric, the simplex schedule or the
/// table size and this goes red — with the regenerated literals written to disk, so
/// the fix is a paste, not a re-derivation. Without it the table is exactly the
/// "typed the same fact twice" hazard that drifts silently.
@MainActor
final class IlluminatoramaLTCTableTests: XCTestCase {

    /// Absolute tolerance per texel component. The fit is deterministic (stratified
    /// sampling, no RNG), so the arms should agree to the last bit in principle —
    /// but the same source compiles at different optimisation levels here vs. in the
    /// generator, and FP contraction is free to differ. 1e-3 is far tighter than any
    /// visible difference in a 16×16 lobe-shape table and still catches a real
    /// change to the fit.
    private let tolerance: Float = 1e-3

    func testBakedTableMatchesTheRealFitter() {
        let baked = IlluminatoramaLTCTable.baked
        let fresh = IlluminatoramaLTC.fitTable(size: IlluminatoramaLTCTable.size)

        XCTAssertEqual(baked.size, fresh.size)
        XCTAssertEqual(baked.mat.count, fresh.size * fresh.size,
                       "baked mat table is the wrong length for its size")
        XCTAssertEqual(baked.mag.count, fresh.size * fresh.size,
                       "baked mag table is the wrong length for its size")
        guard baked.mat.count == fresh.mat.count, baked.mag.count == fresh.mag.count else {
            return   // length assertions above already reported it
        }

        var worst: (label: String, delta: Float, index: Int) = ("", 0, 0)
        for (name, pair) in [("mat", (baked.mat, fresh.mat)), ("mag", (baked.mag, fresh.mag))] {
            for i in pair.0.indices {
                let d = (pair.0[i] - pair.1[i])
                let m = max(max(abs(d.x), abs(d.y)), max(abs(d.z), abs(d.w)))
                if m > worst.delta { worst = (name, m, i) }
            }
        }

        if worst.delta > tolerance {
            let path = NSTemporaryDirectory() + "IlluminatoramaLTCTable.regenerated.swift"
            try? Self.literals(for: fresh).write(toFile: path, atomically: true, encoding: .utf8)
            let cell = "row \(worst.index / fresh.size), col \(worst.index % fresh.size)"
            XCTFail("""
                Baked LTC table no longer matches the fitter — worst \(worst.label) delta \
                \(worst.delta) at \(cell) (tolerance \(tolerance)).
                If the fit changed on purpose, paste the regenerated texels from:
                  \(path)
                into IlluminatoramaLTCTable.swift (and update `maxError` to \(fresh.maxError)).
                """)
        }
    }

    /// The table is only allowed to be TRUSTED if the fit it came from still passes
    /// the brute-force Monte-Carlo check against the actual GGX BRDF — otherwise the
    /// renderer must fall back to most-representative-point specular. Assert the
    /// recorded certificate agrees with a fresh verdict, so a table baked from a
    /// broken fit can't claim validity.
    func testBakedCertificateMatchesAFreshValidation() {
        let fresh = IlluminatoramaLTC.fitTable(size: IlluminatoramaLTCTable.size)
        XCTAssertEqual(IlluminatoramaLTCTable.validated, fresh.validated,
                       "baked validation verdict disagrees with a fresh fit's")
        XCTAssertTrue(fresh.validated,
                      "LTC fit no longer matches brute-force ground truth (max rel err \(fresh.maxError))")
        XCTAssertEqual(IlluminatoramaLTCTable.maxError, fresh.maxError, accuracy: 0.01,
                       "recorded maxError drifted from the fit's")
    }

    /// Guard the shape of the baked data itself: a truncated or mis-shaped literal
    /// block would otherwise only show up as a wrong-looking highlight.
    func testBakedTableIsWellFormed() {
        let t = IlluminatoramaLTCTable.baked
        XCTAssertEqual(t.mat.count, t.size * t.size)
        XCTAssertEqual(t.mag.count, t.size * t.size)
        for v in t.mat { XCTAssertTrue(v.x.isFinite && v.y.isFinite && v.z.isFinite && v.w.isFinite) }
        for v in t.mag {
            XCTAssertTrue(v.x.isFinite && v.y.isFinite, "split-sum terms must be finite")
            XCTAssertGreaterThanOrEqual(v.x, 0, "specular scale can't be negative")
            XCTAssertGreaterThanOrEqual(v.y, 0, "specular bias can't be negative")
        }
    }

    // ── Regeneration ─────────────────────────────────────────────────────────

    /// Emit the two texel arrays exactly as `IlluminatoramaLTCTable.swift` holds
    /// them, so a legitimate fit change is a copy-paste rather than a hunt for
    /// whatever script produced the numbers last time.
    private static func literals(for table: IlluminatoramaLTC.Table) -> String {
        func block(_ name: String, _ values: [SIMD4<Float>]) -> String {
            var s = "    static let \(name): [SIMD4<Float>] = [\n"
            for row in 0..<table.size {
                s += "        // roughness row \(row)\n"
                for col in 0..<table.size {
                    let v = values[row * table.size + col]
                    s += "        SIMD4(\(v.x), \(v.y), \(v.z), \(v.w)),\n"
                }
            }
            return s + "    ]\n"
        }
        return "    public static let maxError: Float = \(table.maxError)\n\n"
            + block("matTexels", table.mat) + "\n" + block("magTexels", table.mag)
    }
}
