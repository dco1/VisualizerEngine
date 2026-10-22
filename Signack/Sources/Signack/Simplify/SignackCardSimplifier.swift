import Foundation
import simd

/// Simplifies FOLIAGE — disconnected leaf cards — by removing whole cards in importance order and
/// enlarging the survivors to hold the region's apparent density. S0's leaf budget, made local:
/// each group of clusters thins on its own, so the near side of a crown can stay full while the far
/// side thins.
///
/// Edge collapse is meaningless here: a card has no neighbours to collapse into, and a part of a
/// lobed fan is a torn leaf. So a group keeps the most important prefix of ITS cards (one global
/// Efraimidis–Spirakis ranking from `SignackImportance` — every prefix is a weighted sample, so the
/// interior thins faster but never empties) that fits the target, and every coarser level's cards
/// are a subset of the finer level's.
///
/// **Coverage.** Survivors scale about their own centre by (keptArea / sourceArea)^−exponent ≤ max,
/// quantised DOWN to `scaleStep`. The quantisation is what keeps memory down: an enlarged card is a
/// new set of vertices, and every card at one quantised scale is welded to ONE copy by the builder.
/// Down, never to nearest: the first level's ~×1.14 snaps to ×1 (no copy at all), and S0 measured
/// that over-compensating reads DENSER far than near — a pop the wrong way. Measured on a 6 000-card
/// test crown: round-to-nearest at 0.1 cost 0.88× extra vertices, floor at 0.15 costs 0.38×.
///
/// **Error.** How much farther apart the survivors stand than the cards they replace —
/// √(sourceArea / kept) − √(sourceArea / sourceCards), in metres. Under a pixel of that, thinning is
/// invisible because the gaps it opens are sub-pixel.
public struct SignackCardSimplifier: SignackSimplifier {
    /// Per card: its position in the importance ranking (0 = kept longest).
    public let rank: [Int]
    public let cards: [SignackCard]
    /// A region never thins below this fraction of the triangles it started with.
    public var minimumFraction: Float = 0.12
    public var coverageExponent: Float = 0.22
    public var maxCoverageScale: Float = 1.4
    public var scaleStep: Float = 0.15

    /// Rank `cards` (atom id = index into `cards`) by `importance`, seeded.
    public init(cards: [SignackCard], importance: SignackImportance = .foliage, seed: UInt64) {
        let ranking = SignackRanking(cards: cards, weights: importance.weights(for: cards), seed: seed)
        var rank = [Int](repeating: 0, count: cards.count)
        for (k, i) in ranking.order.enumerated() { rank[i] = k }
        self.rank = rank
        self.cards = cards
    }

    public func simplify(_ g: SignackGroupInput, mesh: SignackMeshView) -> SignackGroupOutput? {
        guard let atoms = g.atoms else { return nil }
        // This group's cards, most important first, with their triangles.
        var trisOf: [UInt32: [Int]] = [:]
        for (t, a) in atoms.enumerated() { trisOf[a, default: []].append(t) }
        let present = trisOf.keys.sorted { rank[Int($0)] < rank[Int($1)] }
        var kept: [UInt32] = []
        var keptTris = 0
        var keptArea: Float = 0
        for a in present {
            let n = trisOf[a]!.count
            if keptTris + n > g.targetTriangles, !kept.isEmpty { break }
            kept.append(a); keptTris += n; keptArea += cards[Int(a)].area
        }
        guard kept.count < present.count,
              Float(keptTris) >= minimumFraction * Float(g.sourceTriangles) else { return nil }

        let raw = g.sourceArea > 0 ? pow(max(keptArea, 1e-12) / g.sourceArea, -coverageExponent) : 1
        let scale = 1 + ((min(maxCoverageScale, max(1, raw)) - 1) / scaleStep + 1e-4).rounded(.down) * scaleStep

        var out: [SIMD3<UInt32>] = []
        var outAtoms: [UInt32] = []
        var newVerts: [SignackDerivedVertex] = []
        var local: [SignackDerivedVertex: UInt32] = [:]
        for a in kept.sorted() {
            let centre = cards[Int(a)].centroid
            for t in trisOf[a]! {
                let tri = g.triangles[t]
                var r = SIMD3<UInt32>()
                for k in 0..<3 {
                    let source = mesh.origin[Int(tri[k])]
                    if scale == 1 { r[k] = source; continue }
                    let d = SignackDerivedVertex(source: source, center: centre, scale: scale)
                    if let i = local[d] { r[k] = i } else {
                        let i = UInt32(newVerts.count) | SignackGroupOutput.newVertexFlag
                        local[d] = i; newVerts.append(d); r[k] = i
                    }
                }
                out.append(r); outAtoms.append(a)
            }
        }
        let spacing = { (n: Int) in (g.sourceArea / Float(max(n, 1))).squareRoot() }
        let error = max(0, spacing(kept.count) - spacing(g.sourceAtoms))
        return SignackGroupOutput(triangles: out, atoms: outAtoms, newVertices: newVerts, error: error)
    }
}
