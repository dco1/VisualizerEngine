import Foundation
/// A total order over cards, most worth keeping first — a weighted sample WITHOUT replacement.
///
/// Each card draws a seeded key `ln(u) / w` (Efraimidis–Spirakis); sorting by key gives an order
/// in which EVERY PREFIX is itself an importance-weighted sample of the whole set. That is the
/// property the budget ladder rests on: a thinner budget is a shorter prefix of the same order, so
/// every rung's survivors are a subset of the rung above. Moving down a rung only ever REMOVES
/// leaves — nothing new appears, nothing jumps — which is what keeps a distance switch from
/// reading as a pop.
///
/// Weighted sampling rather than a hard sort by weight: a hard sort would strip the interior
/// completely before thinning the shell at all, and a crown seen through its own gaps would be
/// hollow. Here the interior thins faster, never to nothing.
public struct SignackRanking: Sendable {
    /// Card indices, most important first.
    public let order: [Int]
    /// `prefixTriangles[k]` = triangles in the first k cards of `order` (k = 0…count).
    public let prefixTriangles: [Int]

    public init(cards: [SignackCard], weights: [Float], seed: UInt64) {
        precondition(cards.count == weights.count, "one weight per card")
        var keyed: [(key: Double, index: Int)] = []
        keyed.reserveCapacity(cards.count)
        for i in cards.indices {
            let u = SignackHash.unit(seed: seed, index: UInt64(i))
            let w = Double(max(weights[i], 1e-12))
            keyed.append((log(u) / w, i))
        }
        // Larger key (closer to 0) first; ties by index, so the order is total and stable.
        keyed.sort { $0.key != $1.key ? $0.key > $1.key : $0.index < $1.index }
        order = keyed.map(\.index)
        var prefix = [Int](repeating: 0, count: cards.count + 1)
        for (k, i) in order.enumerated() { prefix[k + 1] = prefix[k] + cards[i].triangleCount }
        prefixTriangles = prefix
    }

    /// The longest prefix whose triangles fit `budget` — a pure prefix (never skip-and-fill), so
    /// a smaller budget's cards are always a subset of a larger one's.
    public func cardCount(fitting budget: Int) -> Int {
        guard budget > 0 else { return 0 }
        // Binary search the monotone prefix sums.
        var lo = 0, hi = order.count
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if prefixTriangles[mid] <= budget { lo = mid } else { hi = mid - 1 }
        }
        return lo
    }

    public var totalTriangles: Int { prefixTriangles.last ?? 0 }
}

/// Seeded, platform-stable hashing (SplitMix64) — never `hashValue`, which reseeds per process.
enum SignackHash {
    static func mix(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// A uniform in (0, 1), never 0 or 1, from (seed, index).
    static func unit(seed: UInt64, index: UInt64) -> Double {
        let h = mix(seed ^ mix(index))
        return (Double(h >> 11) + 0.5) / Double(1 << 53)
    }
}
