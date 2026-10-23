import XCTest
@testable import VisualizerMaterials

/// The tile realism pass (`TileRealism`): eased rim, gritty grout, per-tile hue, glaze defects.
/// GPU-free; each claim is a number, not a look.
final class TileRealismStandaloneTests: XCTestCase {
    private let cases = TileRealismPreviewTests.cases

    private func bake(_ p: TileParams, on: Bool) -> MaterialChannels {
        let was = MaterialGenerator.tileRealismEnabled
        MaterialGenerator.tileRealismEnabled = on
        defer { MaterialGenerator.tileRealismEnabled = was }
        return MaterialGenerator.tile(size: 256, params: p, tiling: .librarySwatch)
    }
    private func std(_ v: [Double]) -> Double {
        let m = v.reduce(0, +) / Double(v.count)
        return (v.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(v.count)).squareRoot()
    }

    func testStaysPlausibleAndSeamless() {
        for (name, p) in cases {
            let a = TextureAudit.audit(bake(p, on: true))
            XCTAssertEqual(a.albedoOutOfRange, 0, "\(name) albedo out of range")
            XCTAssertTrue(a.isPlausible, "\(name) not plausible")
            XCTAssertFalse(a.isSeamy, "\(name) seams: \(a.seamScoreX),\(a.seamScoreY)")
        }
    }

    /// A real wrap check, not the audit's ratio: the step ACROSS the seam must be no bigger than
    /// the biggest step anywhere in the adjacent interior.
    func testWrapsAtTheSeam() {
        for (name, p) in cases {
            let ch = bake(p, on: true), n = ch.size
            func lum(_ x: Int, _ y: Int) -> Double { let c = ch.albedo[ch.idx(x, y)]; return c.x + c.y + c.z }
            var seam = 0.0, interior = 0.0
            for y in 0..<n {
                seam = max(seam, abs(lum(0, y) - lum(n - 1, y)))
                for x in 1..<(n - 1) { interior = max(interior, abs(lum(x, y) - lum(x - 1, y))) }
            }
            XCTAssertLessThanOrEqual(seam, interior + 1e-9, "\(name) seam step \(seam) > interior \(interior)")
        }
    }

    func testIsDeterministic() {
        let p = cases[0].1
        XCTAssertEqual(bake(p, on: true).albedo, bake(p, on: true).albedo)
    }

    /// Grout used to be one flat colour; it must now carry grain, and the lever must restore flat.
    func testGroutHasGrain() {
        let p = cases[0].1
        func groutLuma(_ ch: MaterialChannels) -> [Double] {
            // joint texels: the recessed floor of the height field (joint = body − 0.7)
            (0..<ch.albedo.count).filter { ch.height[$0] < 0.45 }.map { let c = ch.albedo[$0]; return c.x + c.y + c.z }
        }
        XCTAssertGreaterThan(std(groutLuma(bake(p, on: true))), 3 * std(groutLuma(bake(p, on: false))) + 0.01)
    }

    /// Glaze batches drift in hue: per-tile R/B ratio must spread, and must have been constant-ish before.
    func testTilesDifferInHueNotJustBrightness() {
        let p = cases[0].1
        func ratioSpread(_ ch: MaterialChannels) -> Double {
            // 4×4 tiles on a 256 bake at 0.3048 m: sample each tile's centre texel
            let n = ch.size, cols = 4
            var r = [Double]()
            for j in 0..<cols { for i in 0..<cols {
                let c = ch.albedo[ch.idx((2 * i + 1) * n / (2 * cols), (2 * j + 1) * n / (2 * cols))]
                r.append(c.x / c.z)
            } }
            return std(r)
        }
        XCTAssertGreaterThan(ratioSpread(bake(p, on: true)), 1.5 * ratioSpread(bake(p, on: false)))
    }

    /// Defects are the exception, not the texture: pinholes + craze + chips darken under 1.5 % of
    /// the face texels, and the lever off removes every one.
    func testDefectsAreSparse() {
        let p = cases[1].1   // gloss shows the full set
        let on = bake(p, on: true), off = bake(p, on: false)
        var changed = 0, face = 0
        for i in 0..<on.albedo.count where off.height[i] > 0.9 {
            face += 1
            if on.albedo[i].x < off.albedo[i].x * 0.85 { changed += 1 }
        }
        XCTAssertLessThan(Double(changed) / Double(face), 0.015)
        XCTAssertGreaterThan(changed, 0, "gloss tile baked no defects at all")
    }

    /// The `.square` edge gets an eased rim: the face falls toward the joint over a band, so the
    /// height just inside the arris sits below the face plateau (it was a flat plateau to the joint).
    func testSquareEdgeIsEased() {
        let p = cases[0].1
        // Height deficit summed over the first four face texels after the joint, against the face
        // plateau (median of the run). Starts below the joint's own floor so the antialias texels count.
        func rimDeficit(_ ch: MaterialChannels) -> Double {
            let n = ch.size, y = n / 8 + n / 16
            var run = [Double]()
            for x in 0..<(n / 4) {
                let h = ch.height[ch.idx(x, y)]
                if run.isEmpty && h < 0.5 { continue }
                if h < 0.35 { break }
                run.append(h)
            }
            guard run.count > 12 else { return 0 }
            let plateau = run.sorted()[run.count / 2]
            return run.prefix(4).reduce(0) { $0 + max(0, plateau - $1) }
        }
        XCTAssertGreaterThan(rimDeficit(bake(p, on: true)), rimDeficit(bake(p, on: false)) + 0.03)
    }

    /// One-time cost only: the pass must not blow up a bake (it ships in the atlas, never per frame).
    func testBakeCostStaysBounded() {
        let p = cases[1].1
        func ms(_ on: Bool) -> Double {
            let was = MaterialGenerator.tileRealismEnabled; MaterialGenerator.tileRealismEnabled = on
            defer { MaterialGenerator.tileRealismEnabled = was }
            let t = Date(); _ = MaterialGenerator.tile(size: 512, params: p, tiling: .librarySwatch)
            return Date().timeIntervalSince(t) * 1000
        }
        _ = ms(true); _ = ms(false)
        XCTAssertLessThan(ms(true), 2.5 * ms(false) + 20)
    }
}
