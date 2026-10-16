import XCTest
@testable import VisualizerRendering

/// **The supersample → output reduction, measured on the CPU** (Daydream DH-1365).
///
/// GPU-free: `ReconstructionModel` pushes analytic scenes (sinusoids, a step, a zone plate, a
/// slanted edge) through the photo lane's sampling — the still's one-internal-pixel jitter box,
/// optionally the TAA still sharpen — and then through `IlluminatoramaReconstructionFilter.gather1D`,
/// the shader's own arithmetic. What it measures:
///
/// * `mtf` — how much of a below-Nyquist sinusoid survives (sharpness);
/// * `alias` — RMS output for 0.5–2 cycles/px input, which a perfect reconstruction would print as
///   nothing (every bit of it is a false, lower-frequency pattern);
/// * `pattern` — the fixed-pattern error a kernel that changes shape from pixel to pixel adds;
/// * step under/overshoot — the dark halo a bright window prints on the wall beside it is the
///   undershoot × (contrast − 1).
///
/// Numbers recorded 2026-10-15 (the table `testPrintTheMeasurementTable` prints):
///
/// | chain (no TAA sharpen)      | MTF@.25 | MTF@.45 | MTF50 c/px | alias | pattern | ZP alias | step under |
/// |---|---|---|---|---|---|---|---|
/// | legacy 4-tap @ 1.0×         | 0.450 | 0.017 | 0.235 | 0.075 | 0      | 0.051 | 0 |
/// | legacy 4-tap @ 1.5× (live)  | 0.745 | 0.357 | 0.376 | 0.113 | 5.0 %  | 0.061 | 0 |
/// | legacy 4-tap @ 1.61× (4K 16:10 export) | 0.784 | 0.430 | 0.411 | 0.130 | 7.3 % | 0.069 | 0 |
/// | legacy 4-tap @ 1.78× (4K on Danny's window) | 0.820 | 0.508 | 0.455 | 0.150 | 6.7 % | 0.086 | 0 |
/// | legacy 4-tap @ 2.0× (= box) | 0.900 | 0.699 | ≥ 0.5 | 0.245 | 0      | 0.204 | 0 |
/// | Blackman–Harris r 1.25 @ 2× | 0.840 | 0.563 | ≈ 0.49 | 0.174 | 0     | 0.118 | 0 |
/// | **Blackman–Harris r 1.5 @ 2×** | 0.786 | 0.451 | 0.421 | 0.121 | 0  | 0.064 | 0 |
/// | Blackman–Harris r 1.75 @ 2× | 0.727 | 0.345 | 0.366 | 0.083 | 0     | 0.031 | 0 |
/// | Mitchell–Netravali r 2 @ 2× | 0.822 | 0.440 | 0.421 | 0.102 | 0     | 0.043 | **1.9 %** |
///
/// Reading it: the still's 2× reduction is a one-pixel box — the sharpest and the most aliased
/// chain in the table. A 4K export does not even get that: `maxInternalMegapixels` drops its scale
/// to 1.6–1.8×, where the legacy kernel changes shape from pixel to pixel (a 5–7 % fixed pattern on
/// fine detail). Blackman–Harris at r 1.5 halves the alias of the 2× box (0.245 → 0.121; zone plate
/// 0.204 → 0.064) at MTF50 0.42 c/px, with no fixed pattern and no ringing at any scale. Mitchell is
/// as sharp for less alias, but its 1.9 % undershoot is a NEAR-BLACK halo beside a 50:1 window
/// (1 − 49 × 0.019 = 7 % of the wall) — in linear HDR a negative lobe is not a subtle sharpening.
/// The width (1.25 / 1.5 / 1.75) is a sharpness-vs-aliasing look call (DH-1365's report flags it).
final class ReconstructionFilterTests: XCTestCase {
    typealias F = IlluminatoramaReconstructionFilter
    typealias M = ReconstructionModel

    static let declared: [F] = [F(kernel: .box, radius: 0.5), F(kernel: .tent, radius: 1),
                                .mitchellNetravali(), .blackmanHarris()]
    static let scales: [Double] = [1.0, 1.3, 1.5, 1.61, 1.78, 2.0, 2.5, 3.0, 4.0]

    // MARK: contract

    func testEveryReductionIsNormalisedAtEveryScaleAndPhase() {
        for f in Self.declared + [.legacy] {
            for s in Self.scales {
                let n = Int((64 * s).rounded(.up))
                for o in 0..<64 {
                    let taps = f.gather1D(outputPixel: o, scale: s, internalCount: n)
                    let sum = taps.reduce(0) { $0 + $1.weight }
                    XCTAssertEqual(sum, 1, accuracy: 1e-12, "\(f.kernel) @ \(s)× pixel \(o)")
                    XCTAssertLessThanOrEqual(taps.count, 2 * F.maxTapHalfWidth + 1)
                }
            }
        }
    }

    func testUniformEncodingKeepsLegacyAtZero() {
        XCTAssertEqual(F.legacy.uniform, .zero, "kernel 0 is the shader's legacy branch — byte-identical frames")
        XCTAssertEqual(F.blackmanHarris(radius: 1.5).uniform, SIMD2(4, 1.5))
        XCTAssertEqual(F.mitchellNetravali().uniform, SIMD2(3, 2))
        for k in F.Kernel.allCases where k != .legacy { XCTAssertGreaterThan(k.rawValue, 0) }
    }

    func testAKernelTooWideForTheTapBudgetIsNarrowedNotTruncated() {
        let f = F.blackmanHarris(radius: 3)
        XCTAssertEqual(f.effectiveRadius(scale: 4), 2, accuracy: 1e-6)
        XCTAssertEqual(f.effectiveRadius(scale: 2), 3, accuracy: 1e-6)
        // Narrowed, the window still ends at ~0: its tails were not cut off.
        let taps = f.gather1D(outputPixel: 40, scale: 4, internalCount: 400)
        let edge = taps.map(\.weight).min() ?? 1
        XCTAssertLessThan(edge, 1e-3)
    }

    // MARK: what the legacy reduction actually is

    func testLegacyReductionIsABoxAtTwoATentAtOneAndPhaseVaryingBetween() {
        // 2.0×: exactly the 2 × 2 box (two texels at ½ each).
        let at2 = F.legacy.gather1D(outputPixel: 10, scale: 2, internalCount: 64)
        let merged2 = Dictionary(at2.map { ($0.texel, $0.weight) }, uniquingKeysWith: +).filter { $0.value > 1e-12 }
        XCTAssertEqual(merged2, [20: 0.5, 21: 0.5])
        // 1.0×: the [¼ ½ ¼] tent — a blur, where the shader comment promised a single read.
        let at1 = F.legacy.gather1D(outputPixel: 10, scale: 1, internalCount: 64)
        let merged1 = Dictionary(at1.map { ($0.texel, $0.weight) }, uniquingKeysWith: +).filter { $0.value > 1e-12 }
        XCTAssertEqual(merged1, [9: 0.25, 10: 0.5, 11: 0.25])
        // 1.78×: the kernel changes per pixel — a fixed pattern on fine detail.
        let legacy = M.summary(.init(filter: .legacy, scale: 1.78))
        let bh = M.summary(.init(filter: .blackmanHarris(), scale: 1.78))
        XCTAssertGreaterThan(legacy.patternRMS, 0.04, "legacy @ 1.78× fixed pattern \(legacy.patternRMS)")
        XCTAssertLessThan(bh.patternRMS, 0.001, "a declared kernel is phase-exact: \(bh.patternRMS)")
        // …and at 1.0× it erases detail the declared kernel keeps.
        let soft = M.summary(.init(filter: .legacy, scale: 1.0))
        XCTAssertLessThan(soft.mtfNyquistish, 0.05)
        XCTAssertLessThan(soft.mtf50, 0.25)
    }

    // MARK: the Maximum default

    func testBlackmanHarrisNeverRingsAndMitchellPrintsADarkHaloBesideAWindow() {
        let bh = F.blackmanHarris()
        for i in 0...300 {
            let x = -1.5 + 3 * Double(i) / 300
            XCTAssertGreaterThanOrEqual(bh.weight(x), 0, "Blackman–Harris is non-negative at \(x)")
        }
        let bhStep = M.stepExcursions(.init(filter: bh, scale: 2))
        XCTAssertLessThan(bhStep.under, 1e-12)
        XCTAssertLessThan(bhStep.over, 1e-12)
        let mn = M.stepExcursions(.init(filter: .mitchellNetravali(), scale: 2))
        XCTAssertGreaterThan(mn.under, 0.015)
        // A window at 50× the wall: the wall pixel beside it keeps only this fraction of its light.
        let wallKept = 1 - 49 * mn.under
        XCTAssertLessThan(wallKept, 0.2, "Mitchell's lobe × a 50:1 window leaves \(wallKept) of the wall")
    }

    func testBlackmanHarrisHalvesTheAliasingOfTheStillsTwoTimesBox() {
        let box = M.Chain(filter: .legacy, scale: 2)
        let bh = M.Chain(filter: .blackmanHarris(), scale: 2)
        let sBox = M.summary(box), sBH = M.summary(bh)
        XCTAssertLessThan(sBH.aliasRMS, 0.55 * sBox.aliasRMS, "alias \(sBox.aliasRMS) → \(sBH.aliasRMS)")
        XCTAssertGreaterThan(sBH.mtf50, 0.40, "and keeps detail: MTF50 \(sBH.mtf50) c/px")
        let zBox = M.zonePlate(box, size: 128), zBH = M.zonePlate(bh, size: 128)
        XCTAssertLessThan(zBH.alias, 0.4 * zBox.alias, "zone plate alias \(zBox.alias) → \(zBH.alias)")
        XCTAssertGreaterThan(zBH.pass, 0.9, "zone plate pass band \(zBH.pass)")
        // Against what a 4K export gets TODAY (legacy at the capped 1.61–1.78×): less alias, no
        // fixed pattern, comparable sharpness.
        for s in [1.61, 1.78] {
            let today = M.summary(.init(filter: .legacy, scale: s))
            XCTAssertLessThan(sBH.aliasRMS, today.aliasRMS, "vs legacy @ \(s)×")
            XCTAssertGreaterThan(sBH.mtf50, today.mtf50 - 0.04, "vs legacy @ \(s)×")
        }
    }

    /// The ISO-12233-style slanted-edge estimator is the instrument the GPU measurement will use on
    /// real frames (DH-1365's remaining work). Prove it on frames whose answer is known.
    func testSlantedEdgeEstimatorAgreesWithTheAnalyticMTF() {
        for chain in [M.Chain(filter: .legacy, scale: 1.5), M.Chain(filter: .blackmanHarris(), scale: 2),
                      M.Chain(filter: F(kernel: .tent, radius: 1), scale: 2)] {
            let analytic = M.summary(chain).mtf50
            let measured = M.slantedEdge(chain).mtf50
            XCTAssertEqual(measured, analytic, accuracy: 0.01, "\(chain.filter.kernel) @ \(chain.scale)×")
        }
    }

    // MARK: shader mirror

    func testShaderSourceMirrorsTheSwiftKernelCoefficients() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaTonemap.metal")
        let src = try String(contentsOf: url, encoding: .utf8)
        let a = F.blackmanHarrisA
        for c in [a.0, a.1, a.2, a.3] {
            XCTAssertTrue(src.contains(String(format: "%.5f", c)), "BH coefficient \(c) missing from the shader")
        }
        XCTAssertTrue(src.contains("kReconMaxHalf = \(F.maxTapHalfWidth);"))
        XCTAssertTrue(src.contains("const float B = 1.0 / 3.0, C = 1.0 / 3.0;"))
        XCTAssertEqual(F.mitchellB, 1.0 / 3.0); XCTAssertEqual(F.mitchellC, 1.0 / 3.0)
        for k in F.Kernel.allCases where k != .legacy {
            XCTAssertTrue(src.contains("kern == \(k.rawValue)u"), "shader has no branch for \(k)")
        }
        XCTAssertTrue(src.contains("frame.taaJitterDelta.z > 0.5"), "the branch is gated on the zw slot")
    }

    // MARK: the record

    func testPrintTheMeasurementTable() {
        let chains: [(String, M.Chain)] = [
            ("legacy @1.0", .init(filter: .legacy, scale: 1.0)),
            ("legacy @1.5", .init(filter: .legacy, scale: 1.5)),
            ("legacy @1.61", .init(filter: .legacy, scale: 1.61)),
            ("legacy @1.78", .init(filter: .legacy, scale: 1.78)),
            ("legacy @2.0", .init(filter: .legacy, scale: 2.0)),
            ("BH r1.25 @2", .init(filter: .blackmanHarris(radius: 1.25), scale: 2)),
            ("BH r1.5 @2", .init(filter: .blackmanHarris(), scale: 2)),
            ("BH r1.75 @2", .init(filter: .blackmanHarris(radius: 1.75), scale: 2)),
            ("MN r2 @2", .init(filter: .mitchellNetravali(), scale: 2)),
            ("BH r1.5 @1.0", .init(filter: .blackmanHarris(), scale: 1.0)),
        ]
        print("DH-1365 reconstruction  sharpen  MTF.25  MTF.45  MTF50  alias  pattern  under%  ZPpass  ZPalias  edgeMTF50")
        for (name, base) in chains {
            for sharpen in [false, true] {
                var c = base; c.sharpen = sharpen
                let s = M.summary(c), st = M.stepExcursions(c), z = M.zonePlate(c, size: 128), e = M.slantedEdge(c)
                print(String(format: "DH-1365 %-14@  %@  %.3f  %.3f  %.3f  %.3f  %.4f  %.2f  %.3f  %.3f  %.3f",
                             name as NSString, sharpen ? "Y" : "-", s.mtfQuarter, s.mtfNyquistish, s.mtf50,
                             s.aliasRMS, s.patternRMS, st.under * 100, z.pass, z.alias, e.mtf50))
            }
        }
    }
}
