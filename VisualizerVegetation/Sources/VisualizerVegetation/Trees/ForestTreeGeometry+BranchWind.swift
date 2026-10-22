import simd

/// Signack S5 (Daydream DH-1461) — the bake half of the per-branch (limb) wind: Pivot-Painter for
/// the Forest soup. Every primary limb is recorded as it grows (`openLimb` / `closeLimb` around
/// its whole recursion — its wood, sub-branches, twigs and the leaf cards hung on them are one
/// contiguous vertex range, because every vertex goes through `tri()`), then `bakeBranchWind()`
/// writes into each vertex's uv the record the renderer's limb layer reads
/// (VisualizerRendering `IlluminatoramaBranchWind.h`):
///
///     uv = (20 + b, φ)    b — weight along the limb: (distance from the limb's base / the limb's
///                             reach)^1.3, 0 at the joint so the limb never tears off its parent
///                         φ — the limb's phase, one per limb, hashed from the tree seed (the
///                             generator's RNG stream is never touched, so geometry is unchanged)
///
/// A leaf CARD carries one b (its centroid's), so it moves rigidly with its branch. Cards emitted
/// outside any limb (the volumetric crown fill, junction tufts) join the nearest limb by distance
/// to its base→tip segment, so the whole crown moves with its limbs instead of shearing against
/// them. Trunk and leader wood keep uv = 0 (trunk sway only).
extension ForestTreeGeometry.Soup {

    /// One primary limb: its base (the pivot), its phase, and its vertex range.
    public struct Limb: Sendable {
        public var pivot: SIMD3<Float>
        public var phase: Float
        public var firstVertex: Int
        public var endVertex: Int
    }

    /// The uv band the renderer reads as a limb record (`kBranchWindMarker`).
    public static let branchWindMarker: Float = 20

    /// Start recording a primary limb based at `pivot`. Returns its index, or −1 when the soup has
    /// not opted in (`branchWind == false`).
    public mutating func openLimb(pivot: SIMD3<Float>, seed: UInt64) -> Int {
        guard branchWind else { return -1 }
        var z = seed &+ UInt64(limbs.count) &* 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        let phase = Float(z >> 40) / Float(1 << 24) * 2 * .pi
        limbs.append(Limb(pivot: pivot, phase: phase, firstVertex: positions.count, endVertex: positions.count))
        return limbs.count - 1
    }

    /// Close limb `i` (from `openLimb`) at the current end of the vertex arrays.
    public mutating func closeLimb(_ i: Int) {
        guard i >= 0, i < limbs.count else { return }
        limbs[i].endVertex = positions.count
    }

    /// Write every tree vertex's limb record into `uvs`. No-op unless `branchWind` recorded limbs.
    public mutating func bakeBranchWind() {
        guard branchWind, !limbs.isEmpty, uvs.count == positions.count else { return }
        // Each limb's reach (farthest own vertex from its base) and tip (that vertex).
        var reach = [Float](repeating: 1e-3, count: limbs.count)
        var tip = limbs.map(\.pivot)
        for (li, l) in limbs.enumerated() where l.endVertex > l.firstVertex {
            for v in l.firstVertex..<l.endVertex {
                let d = simd_distance(positions[v], l.pivot)
                if d > reach[li] { reach[li] = d; tip[li] = positions[v] }
            }
        }
        let triCount = indices.count / 3
        // Owner limb of every vertex (−1 = none), from the recorded ranges.
        var owner = [Int32](repeating: -1, count: positions.count)
        for (li, l) in limbs.enumerated() where l.endVertex > l.firstVertex {
            for v in l.firstVertex..<l.endVertex { owner[v] = Int32(li) }
        }
        // Card centroids, and each card's limb: its own if it grew on one, else the nearest.
        var cardSum: [Int32: (SIMD3<Float>, Int, Int32)] = [:]
        for t in 0..<min(triCount, triCard.count) where triCard[t] >= 0 {
            let i0 = Int(indices[t * 3])
            var e = cardSum[triCard[t]] ?? (.zero, 0, owner[i0])
            for k in 0..<3 { e.0 += positions[Int(indices[t * 3 + k])]; e.1 += 1 }
            cardSum[triCard[t]] = e
        }
        func nearestLimb(_ p: SIMD3<Float>) -> Int {
            var best = 0, bestD = Float.greatestFiniteMagnitude
            for (li, l) in limbs.enumerated() {
                let ab = tip[li] - l.pivot
                let s = simd_clamp(simd_dot(p - l.pivot, ab) / max(simd_length_squared(ab), 1e-8), 0, 1)
                let d = simd_distance_squared(p, l.pivot + ab * s)
                if d < bestD { bestD = d; best = li }
            }
            return best
        }
        func record(_ li: Int, at p: SIMD3<Float>) -> SIMD2<Float> {
            let b = pow(min(simd_distance(p, limbs[li].pivot) / reach[li], 1), 1.3)
            return SIMD2(Self.branchWindMarker + b, limbs[li].phase)
        }
        var cardRecord: [Int32: SIMD2<Float>] = [:]
        for (card, e) in cardSum where e.1 > 0 {
            let c = e.0 / Float(e.1)
            cardRecord[card] = record(e.2 >= 0 ? Int(e.2) : nearestLimb(c), at: c)
        }
        // Only geometry that already rides the tree wind (tangent.w 1 leaf / 2–4 bark); never the
        // ground, logs or props a soup may also carry (w 0 / 5 / 6).
        func treeVertex(_ v: Int) -> Bool {
            v < tangents.count && tangents[v].w >= 0.5 && tangents[v].w <= 4.5
        }
        for t in 0..<triCount {
            let card = t < triCard.count ? triCard[t] : -1
            for k in 0..<3 {
                let v = Int(indices[t * 3 + k])
                guard treeVertex(v) else { continue }
                if card >= 0, let r = cardRecord[card] {
                    uvs[v] = r
                } else if owner[v] >= 0 {
                    uvs[v] = record(Int(owner[v]), at: positions[v])
                }
            }
        }
    }
}
