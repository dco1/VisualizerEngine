/// Turn an unindexed triangle soup (three fresh vertices per triangle) into an indexed mesh by
/// merging vertices whose every attribute is bit-identical.
///
/// Generic over the host's vertex type: the host supplies a `Hashable` key per soup vertex (its
/// attributes' bit patterns) and gets back one index per soup vertex into the welded set. Exact
/// equality only — no epsilon — so welding never changes a single shaded value; it only stops a
/// mesh paying for the same vertex once per triangle that touches it. Deterministic: welded
/// vertices are numbered in first-seen order.
public enum SignackVertexWeld {
    public struct Result: Sendable {
        /// For each input vertex, its index in the welded vertex list.
        public var remap: [UInt32]
        /// Input vertex index that became each welded vertex (first occurrence).
        public var representatives: [Int]
    }

    public static func weld<Key: Hashable>(count: Int, key: (Int) -> Key) -> Result {
        var seen: [Key: UInt32] = [:]
        seen.reserveCapacity(count / 2)
        var remap = [UInt32](repeating: 0, count: count)
        var reps: [Int] = []
        reps.reserveCapacity(count / 2)
        for i in 0..<count {
            let k = key(i)
            if let j = seen[k] {
                remap[i] = j
            } else {
                let j = UInt32(reps.count)
                seen[k] = j
                reps.append(i)
                remap[i] = j
            }
        }
        return Result(remap: remap, representatives: reps)
    }
}
