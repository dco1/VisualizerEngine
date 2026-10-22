import Foundation

/// The ladder of budgets a mesh is baked at: rung 0 for the camera's doorstep, each later rung a
/// fraction of it for farther away. Triangle counts, not distances — WHICH rung a frame draws is
/// `SignackLODSelector`'s call, from projected size.
public struct SignackBudgetLadder: Sendable, Equatable {
    /// Hard ceiling on rung 0's TOTAL triangles (fixed + budgeted). A safety bound for
    /// pathological inputs, not a quality knob: set it above any real mesh and rung 0 keeps
    /// everything.
    public var ceiling: Int
    /// Each rung's budgeted triangles as a fraction of rung 0's, descending. First must be 1.
    public var ratios: [Float]
    /// How much of the lost leaf AREA a thinner rung wins back by enlarging its surviving cards:
    /// scale = (keptArea / rung0Area)^(−exponent). 0.5 would conserve area exactly; leaves
    /// overlap, so coverage saturates before area does, and a partial exponent holds the crown's
    /// apparent density without the far rungs turning into plates. 0.22 / 1.4 is MEASURED: at
    /// 0.35 / 1.6 the 15 % rung read DENSER than the full crown (upper-crown leaf coverage 0.504
    /// vs 0.469 on the hero oak), which would be a pop the other way.
    public var coverageExponent: Float
    /// Upper bound on that enlargement.
    public var maxCoverageScale: Float

    public init(ceiling: Int, ratios: [Float] = [1, 0.4, 0.15],
                coverageExponent: Float = 0.22, maxCoverageScale: Float = 1.4) {
        precondition(!ratios.isEmpty && ratios[0] == 1, "rung 0 is the full budget")
        self.ceiling = ceiling
        self.ratios = ratios
        self.coverageExponent = coverageExponent
        self.maxCoverageScale = maxCoverageScale
    }
}

/// S0's answer: which cards each rung keeps. One ranking, many prefixes.
public struct SignackLeafPlan: Sendable {
    public struct Rung: Sendable, Equatable {
        /// The rung keeps the first `cardCount` cards of the ranking.
        public var cardCount: Int
        /// Budgeted triangles the rung keeps (fixed triangles not included).
        public var triangles: Int
        /// Surviving cards are scaled by this about their own centroid (1 on rung 0).
        public var coverageScale: Float
        /// Fraction of rung 0's card AREA the rung keeps (before `coverageScale`).
        public var keptAreaFraction: Float
    }

    public let ranking: SignackRanking
    /// `rankOf[card]` = that card's position in the ranking (0 = most important).
    public let rankOf: [Int]
    public let rungs: [Rung]

    /// Does `rung` keep `card`?
    @inlinable public func keeps(card: Int, rung: Int) -> Bool {
        rankOf[card] < rungs[rung].cardCount
    }
}

/// S0 — spend a triangle budget by importance, not evenly.
public enum SignackLeafBudget {

    /// Rank `cards`, then cut every rung of `ladder` from that one ranking.
    ///
    /// - Parameters:
    ///   - fixedTriangles: triangles that are always kept and count against the ceiling (a tree's
    ///     wood armature).
    ///   - importance: the weighting policy (`.foliage`, or `.uniform` for the even control arm).
    ///   - seed: the mesh's own seed — same seed, same plan, every run.
    public static func plan(cards: [SignackCard], fixedTriangles: Int,
                            ladder: SignackBudgetLadder,
                            importance: SignackImportance = .foliage,
                            seed: UInt64) -> SignackLeafPlan {
        let weights = importance.weights(for: cards)
        let ranking = SignackRanking(cards: cards, weights: weights, seed: seed)
        var rankOf = [Int](repeating: 0, count: cards.count)
        for (k, i) in ranking.order.enumerated() { rankOf[i] = k }

        let rung0Budget = min(ranking.totalTriangles, max(0, ladder.ceiling - fixedTriangles))
        // Area kept by a prefix of the ranking, for coverage compensation.
        var prefixArea = [Double](repeating: 0, count: cards.count + 1)
        for (k, i) in ranking.order.enumerated() { prefixArea[k + 1] = prefixArea[k] + Double(cards[i].area) }

        var rungs: [SignackLeafPlan.Rung] = []
        var rung0Area = 0.0
        for (r, ratio) in ladder.ratios.enumerated() {
            let budget = r == 0 ? rung0Budget : Int(Float(rung0Budget) * ratio)
            let n = ranking.cardCount(fitting: budget)
            let area = prefixArea[n]
            if r == 0 { rung0Area = area }
            var scale: Float = 1
            if r > 0, area > 0, rung0Area > 0 {
                let kept = Float(area / rung0Area)
                scale = min(ladder.maxCoverageScale, max(1, pow(kept, -ladder.coverageExponent)))
            }
            rungs.append(.init(cardCount: n, triangles: ranking.prefixTriangles[n], coverageScale: scale,
                               keptAreaFraction: rung0Area > 0 ? Float(area / rung0Area) : 1))
        }
        return SignackLeafPlan(ranking: ranking, rankOf: rankOf, rungs: rungs)
    }
}
