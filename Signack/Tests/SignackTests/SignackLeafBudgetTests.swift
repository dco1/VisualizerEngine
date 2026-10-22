import XCTest
import simd
@testable import Signack

/// S0 — the leaf budget's contract, GPU-free.
final class SignackLeafBudgetTests: XCTestCase {

    /// A synthetic crown: cards scattered through a 3 m ball, mixed 2- and 20-triangle cards.
    static func crown(count: Int = 4000, seed: UInt64 = 7) -> [SignackCard] {
        var cards: [SignackCard] = []
        var t = 0
        for i in 0..<count {
            let a = SignackHash.unit(seed: seed, index: UInt64(i * 3))
            let b = SignackHash.unit(seed: seed, index: UInt64(i * 3 + 1))
            let c = SignackHash.unit(seed: seed, index: UInt64(i * 3 + 2))
            let theta = Float(a) * 2 * .pi, z = Float(b) * 2 - 1
            let r = 3 * Float(c).squareRoot()
            let s = (1 - z * z).squareRoot()
            let p = SIMD3<Float>(cos(theta) * s, z, sin(theta) * s) * r + SIMD3(0, 5, 0)
            let tris = i % 5 == 0 ? 20 : 2
            cards.append(SignackCard(firstTriangle: t, triangleCount: tris, centroid: p,
                                     normal: simd_normalize(SIMD3(Float(a) - 0.5, 0.3, Float(b) - 0.5)),
                                     area: 0.01))
            t += tris
        }
        return cards
    }

    func testRungsAreNestedAndUnderBudget() {
        let cards = Self.crown()
        let total = cards.reduce(0) { $0 + $1.triangleCount }
        let plan = SignackLeafBudget.plan(cards: cards, fixedTriangles: 1000,
                                          ladder: SignackBudgetLadder(ceiling: 1000 + total / 2), seed: 1)
        XCTAssertLessThanOrEqual(plan.rungs[0].triangles, total / 2)
        for r in 1..<plan.rungs.count {
            XCTAssertLessThanOrEqual(plan.rungs[r].cardCount, plan.rungs[r - 1].cardCount)
            for i in cards.indices where plan.keeps(card: i, rung: r) {
                XCTAssertTrue(plan.keeps(card: i, rung: r - 1), "rung \(r) keeps a card rung \(r - 1) dropped")
            }
            XCTAssertGreaterThanOrEqual(plan.rungs[r].coverageScale, 1)
        }
    }

    func testDeterministic() {
        let cards = Self.crown()
        let l = SignackBudgetLadder(ceiling: 10_000)
        let a = SignackLeafBudget.plan(cards: cards, fixedTriangles: 0, ladder: l, seed: 42)
        let b = SignackLeafBudget.plan(cards: cards, fixedTriangles: 0, ladder: l, seed: 42)
        XCTAssertEqual(a.ranking.order, b.ranking.order)
        XCTAssertEqual(a.rungs, b.rungs)
    }

    /// The shell is kept preferentially, the interior thinned but not emptied.
    func testShellOutlivesInteriorButInteriorSurvives() {
        let cards = Self.crown()
        let plan = SignackLeafBudget.plan(cards: cards, fixedTriangles: 0,
                                          ladder: SignackBudgetLadder(ceiling: 4000), seed: 3)
        let shell = SignackCrownShell(cards: cards)
        var outK = 0, outN = 0, inK = 0, inN = 0
        for i in cards.indices {
            let d = shell.shellDepth(cards[i].centroid)
            let kept = plan.keeps(card: i, rung: 0)
            if d > 0.85 { outN += 1; if kept { outK += 1 } }
            if d < 0.5 { inN += 1; if kept { inK += 1 } }
        }
        let outRate = Float(outK) / Float(outN), inRate = Float(inK) / Float(inN)
        XCTAssertGreaterThan(outRate, 2 * inRate, "shell \(outRate) vs interior \(inRate)")
        XCTAssertGreaterThan(inK, 0, "interior emptied — a crown seen through a gap would be hollow")
    }

    func testSelectorHysteresis() {
        let sel = SignackLODSelector()
        let rungs = [300_000, 120_000, 45_000]
        let near = sel.rung(rungTriangles: rungs, projectedRadiusPixels: 400, previous: nil)
        XCTAssertEqual(near, 0)
        let far = sel.rung(rungTriangles: rungs, projectedRadiusPixels: 60, previous: nil)
        XCTAssertEqual(far, 2)
        // Just past rung 1's threshold: from rung 0 it must not thin yet (hysteresis) …
        let r1 = (Float(rungs[1]) / (1.5 * .pi)).squareRoot() * 0.97
        XCTAssertEqual(sel.rung(rungTriangles: rungs, projectedRadiusPixels: r1, previous: 0), 0)
        // … while a frame already on rung 1 stays there.
        XCTAssertEqual(sel.rung(rungTriangles: rungs, projectedRadiusPixels: r1, previous: 1), 1)
    }

    func testWeldMergesOnlyIdenticalVertices() {
        let keys = [1, 2, 3, 3, 2, 4]
        let w = SignackVertexWeld.weld(count: keys.count) { keys[$0] }
        XCTAssertEqual(w.remap, [0, 1, 2, 2, 1, 3])
        XCTAssertEqual(w.representatives, [0, 1, 2, 5])
    }
}
