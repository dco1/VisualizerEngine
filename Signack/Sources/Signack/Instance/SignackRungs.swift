import simd

/// **S6 — a baked type's LOD RUNGS for batched instances.** Uniform cuts of a `SignackHierarchy`
/// at rising object-space tolerances, each roughly half the triangles of the one before, packed
/// back to back into ONE index list.
///
/// Why rungs and not S2's per-cluster cut, for a planting: a per-cluster cut is per INSTANCE (its
/// own compacted index list and its own draw), which is right for a hero tree and wrong for 800
/// shrubs — 800 index lists, 800 draws per pass. A rung is a cut every instance at that distance
/// can SHARE, so a whole species draws as one instanced draw per rung (`SignackGPUInstanceBatch`
/// picks each instance's rung on the GPU). Each rung is still a crack-free DAG cut
/// (`SignackSelection.cut(_:tolerance:)`); a plant is a few metres across, so one LOD per plant
/// costs nothing a per-region cut would save.
public struct SignackRungs: Sendable, Equatable {
    public struct Rung: Sendable, Equatable {
        /// Object-space error this rung shows (0 = the source mesh).
        public var error: Float
        /// Its run in `indices` (first index, index count).
        public var indexStart: Int
        public var indexCount: Int
        public var triangleCount: Int { indexCount / 3 }

        public init(error: Float, indexStart: Int, indexCount: Int) {
            self.error = error; self.indexStart = indexStart; self.indexCount = indexCount
        }
    }

    public let rungs: [Rung]
    /// Every rung's triangles, concatenated — indexes the hierarchy's vertex buffer.
    public let indices: [UInt32]

    /// Rungs a host built itself (DH-1470 — a planted tree appends far rungs below its DAG's floor):
    /// `rungs` index `indices`, finest first, errors rising.
    public init(rungs: [Rung], indices: [UInt32]) {
        precondition(rungs.allSatisfy { $0.indexStart >= 0 && $0.indexStart + $0.indexCount <= indices.count },
                     "a rung outside its index list")
        self.rungs = rungs
        self.indices = indices
    }

    /// Up to `maxRungs` rungs, each at most `ratio` × the previous rung's triangles. Stops early when
    /// the DAG can get no coarser (its roots).
    public init(_ h: SignackHierarchy, maxRungs: Int = 4, ratio: Float = 0.5) {
        let errors = Set(h.clusters.map(\.lodError).filter { $0.isFinite }).sorted()
        var rungs: [Rung] = []
        var indices: [UInt32] = []
        var lastTris = Int.max
        for e in errors where rungs.count < max(1, maxRungs) {
            let cut = SignackSelection.cut(h, tolerance: e)
            let tris = h.triangleCount(cut)
            guard rungs.isEmpty || Float(tris) <= ratio * Float(lastTris) else { continue }
            rungs.append(Rung(error: e, indexStart: indices.count, indexCount: tris * 3))
            indices.append(contentsOf: h.flatten(cut))
            lastTris = tris
        }
        self.rungs = rungs
        self.indices = indices
    }
}
