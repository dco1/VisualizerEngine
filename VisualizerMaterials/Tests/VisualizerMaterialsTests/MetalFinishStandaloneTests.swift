import XCTest
@testable import VisualizerMaterials

/// **Every metal in the library bakes without a hidden rib** (Daydream DH-0965).
///
/// A range finished in Aluminum read as corrugated sheet: its brush was
/// `fbmTiled(u · 40, v · 3, baseCells: 8)`, which tiles one 8-cell strip forty times across the
/// bake — an exactly periodic 25 mm rib on a 1 m millwork tile — drawn into the ALBEDO, turned
/// into relief by the normal derivation, and streaked across the grain tangent it declared. Brushed
/// Nickel had the same recipe at 15.6 mm; Brushed Steel and Corten carried the periodic stripe
/// quieter; Matte Black's coat term repeated 3 × 3. `PatternAudit` measures each defect as a
/// number, and this gate holds every `.metal` generator to it.
final class MetalFinishStandaloneTests: XCTestCase {

    static let size = 192

    /// Every `.metal` generator the engine ships, by the id Daydream's registry offers it under.
    static let metals: [(String, @Sendable (Int) -> MaterialChannels)] = [
        ("brushed-steel", { MaterialGenerator.brushedSteel(size: $0) }),
        ("aluminum", { MaterialGenerator.aluminum(size: $0) }),
        ("brushed-nickel", { MaterialGenerator.brushedNickel(size: $0) }),
        ("brass", { MaterialGenerator.brass(size: $0) }),
        ("copper", { MaterialGenerator.copper(size: $0) }),
        ("bronze", { MaterialGenerator.bronze(size: $0) }),
        ("chrome", { MaterialGenerator.chrome(size: $0) }),
        ("matte-black", { MaterialGenerator.matteBlack(size: $0) }),
        ("corten", { MaterialGenerator.corten(size: $0) }),
    ]

    /// The brushed family: streaked along a declared grain.
    static let brushed: Set<String> = ["brushed-steel", "aluminum", "brushed-nickel"]

    /// No metal repeats inside its own tile — the corrugation tell.
    func testNoMetalRepeatsInsideItsTile() {
        var report: [String] = []
        for (id, make) in Self.metals {
            let r = PatternAudit.repeatScore(make(Self.size))
            report.append("\(id): u=\(fmt(r.u)) v=\(fmt(r.v))")
            XCTAssertLessThan(max(r.u, r.v), PatternAudit.repeatCeiling,
                              "\(id) repeats inside its tile (u \(fmt(r.u)), v \(fmt(r.v)))")
        }
        print("REPEAT  " + report.joined(separator: "  |  "))
    }

    /// A brushed metal is streaked ALONG the grain it hands the anisotropic lobe, and clearly so —
    /// in BOTH places the brush lives: the base bake's gloss streak (roughness), and the
    /// detail-band hairline (its normal map), which is where the visible brush is drawn.
    func testBrushedMetalsStreakAlongTheirGrain() {
        var report: [String] = []
        for (id, make) in Self.metals where Self.brushed.contains(id) {
            let ch = make(Self.size)
            let base = PatternAudit.elongation(ch.roughness, size: ch.size)
            let hair = PatternAudit.elongation(normals: ch.detailNormal ?? [])
            let agreeBase = PatternAudit.grainAgreement(ch, axis: base.axis)
            let agreeHair = PatternAudit.grainAgreement(ch, axis: hair.axis)
            report.append("\(id): base agree=\(fmt(agreeBase ?? -1)) str=\(fmt(base.strength))"
                          + " hair agree=\(fmt(agreeHair ?? -1)) str=\(fmt(hair.strength))")
            XCTAssertNotNil(agreeBase, "\(id) must declare a grain tangent")
            XCTAssertGreaterThan(agreeBase ?? 0, 0.95, "\(id)'s gloss streak crosses its own grain")
            XCTAssertGreaterThan(base.strength, 0.5, "\(id)'s gloss streak is not directional")
            XCTAssertNotNil(ch.detailNormal, "\(id) must carry its hairline in the detail band")
            XCTAssertGreaterThan(agreeHair ?? 0, 0.95, "\(id)'s hairline crosses its own grain")
            XCTAssertGreaterThan(hair.strength, 0.8, "\(id)'s hairline is not a hairline")
        }
        print("GRAIN  " + report.joined(separator: "  |  "))
    }

    /// A metal's albedo IS its F0: a brushed metal's brush lives in roughness and height, never in
    /// the colour of the steel (the 2026-08-11 "tan corduroy" correction, now held for the family).
    func testBrushedMetalsKeepTheBrushOutOfTheAlbedo() {
        for (id, make) in Self.metals where Self.brushed.contains(id) {
            let ch = make(Self.size)
            let luma = ch.albedo.map { 0.2126 * $0.x + 0.7152 * $0.y + 0.0722 * $0.z }
            let m = luma.reduce(0, +) / Double(luma.count)
            let sd = (luma.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(luma.count)).squareRoot()
            print("ALBEDO  \(id): cv=\(fmt(sd / m))")
            XCTAssertLessThan(sd / m, 0.005, "\(id) paints its brush into the albedo (cv \(fmt(sd / m)))")
        }
    }

    /// Corten's weathering is rain washing oxide DOWN the face — V is world-vertical on every
    /// wall-facing surface — so its runs are long in V.
    func testCortenRunsAreVertical() {
        let ch = MaterialGenerator.corten(size: Self.size)
        let luma = ch.albedo.map { 0.2126 * $0.x + 0.7152 * $0.y + 0.0722 * $0.z }
        let e = PatternAudit.elongation(luma, size: ch.size)
        print("CORTEN  axis=(\(fmt(e.axis.x)), \(fmt(e.axis.y))) strength=\(fmt(e.strength))")
        XCTAssertGreaterThan(abs(e.axis.y), 0.9, "corten's rain runs are not vertical")
    }

    /// The instrument itself: it must see the trap it exists for, and pass what it should.
    func testPatternAuditSeesAScaledCoordinateRepeat() {
        let n = 128
        var periodic = [Double](repeating: 0, count: n * n), random = periodic
        for y in 0..<n {
            for x in 0..<n {
                let u = Double(x) / Double(n), v = Double(y) / Double(n)
                periodic[y * n + x] = Noise.fbmTiled(u * 8, v, baseCells: 4, octaves: 2, seed: 9)
                random[y * n + x] = Noise.fbmTiled(u, v, baseCells: 32, octaves: 2, seed: 9)
            }
        }
        XCTAssertGreaterThan(PatternAudit.repeatScore(periodic, size: n, axis: 0), 0.9)
        XCTAssertLessThan(PatternAudit.repeatScore(random, size: n, axis: 0), PatternAudit.repeatCeiling)
        XCTAssertLessThan(PatternAudit.repeatScore(random, size: n, axis: 1), PatternAudit.repeatCeiling)
    }

    private func fmt(_ x: Double) -> String { String(format: "%.3f", x) }
}
