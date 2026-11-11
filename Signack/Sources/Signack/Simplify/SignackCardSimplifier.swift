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
/// **Coverage — a simpler level may not be a barer one.** Survivors scale about their own centre
/// until the group carries the leaf AREA it stands for: s = √(sourceArea / keptArea), quantised DOWN
/// to `scaleStep`. Area is what the eye reads as cover (N cards of area a hide what N/2 cards of
/// area 2a hide), so the exponent is ½ exactly. It used to be 0.22, capped at ×1.4: half the cards
/// then carried 0.66 of the area, a quarter 0.42, an eighth 0.24 — every crown went bare as the
/// camera pulled away (measured on every species, DH-1658).
///
/// Two limits, and a level that cannot hold `minimumCoverage` inside them is NOT MADE (the group's
/// clusters become roots — the crown simply stops getting simpler there):
/// - `maxCoverageScale`: beyond it a card reads as a plate, not a leaf;
/// - `maxReachGrowth`: enlarging may push a card's edge outward by at most this fraction of the
///   crown's radius, so the crown keeps its SIZE. Leaves are tiny against a crown and never meet
///   it; a palm's frond is a card half the crown long, and may hardly grow at all — so a palm
///   does not thin (it has few triangles to save, and a frond removed is a gap, not a detail).
///
/// The quantisation keeps memory down: an enlarged card is a new set of vertices, and every card
/// at one quantised scale is welded to ONE copy by the builder.
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
    public var coverageExponent: Float = 0.5
    public var maxCoverageScale: Float = 3
    public var scaleStep: Float = 0.1
    /// A level must carry at least this fraction of the leaf area it stands for, or it is not made.
    public var minimumCoverage: Float = 0.8
    /// Most an enlarged card's edge may move outward, as a fraction of the crown's radius.
    public var maxReachGrowth: Float = 0.12
    /// Half the diagonal of the box round every card's centre — the crown's size.
    public let crownRadius: Float

    /// Rank `cards` (atom id = index into `cards`) by `importance`, seeded.
    public init(cards: [SignackCard], importance: SignackImportance = .foliage, seed: UInt64) {
        let ranking = SignackRanking(cards: cards, weights: importance.weights(for: cards), seed: seed)
        var rank = [Int](repeating: 0, count: cards.count)
        for (k, i) in ranking.order.enumerated() { rank[i] = k }
        self.rank = rank
        self.cards = cards
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for c in cards { lo = simd_min(lo, c.centroid); hi = simd_max(hi, c.centroid) }
        crownRadius = cards.isEmpty ? 0 : simd_length(hi - lo) * 0.5
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

        // How far the kept cards reach from their own centres (area-weighted mean of each card's
        // farthest source vertex): what enlarging them pushes outward.
        var reach: Float = 0
        for a in kept {
            let c = cards[Int(a)]
            var r2: Float = 0
            for t in trisOf[a]! {
                for k in 0..<3 { r2 = max(r2, simd_distance_squared(mesh.positions[Int(mesh.origin[Int(g.triangles[t][k])])], c.centroid)) }
            }
            reach += r2.squareRoot() * c.area
        }
        reach /= max(keptArea, 1e-12)
        let raw = g.sourceArea > 0 ? pow(max(keptArea, 1e-12) / g.sourceArea, -coverageExponent) : 1
        let limit = min(maxCoverageScale, 1 + maxReachGrowth * crownRadius / max(reach, 1e-6))
        let scale = 1 + ((min(limit, max(1, raw)) - 1) / scaleStep + 1e-4).rounded(.down) * scaleStep
        guard g.sourceArea <= 0 || keptArea * scale * scale >= minimumCoverage * g.sourceArea else { return nil }

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
