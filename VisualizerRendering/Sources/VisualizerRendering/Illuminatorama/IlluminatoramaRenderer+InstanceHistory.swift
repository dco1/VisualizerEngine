import Metal

// ── Each draw's own last-frame record ─────────────────────────────────────────
//
// The G-buffer's motion vectors read `prevInstances[iid]` at the slot `instances[iid]` sits in, and
// the grouped layout (`uploadInstances`) puts each mesh kind in one contiguous run. A frame where a
// kind comes or goes moves every later run, so before this each of those draws reprojected with a
// neighbour's matrix: one frame of smear on everything after it in the list (Dad Hat Factory's live
// gallery, a hat scrolling on screen), which hosts covered by throwing the whole history away.
//
// Now: each slot gets a key (`InstanceSlotKey`). When this frame's keys differ from last frame's,
// last frame's records are copied into a buffer in THIS frame's layout and bound as the previous
// instances for the frame. Only a frame whose make-up changed pays (one dictionary build and one
// pass over the records); a steady list costs nothing here (its keys are last frame's array, `InstanceLayout`).

extension IlluminatoramaRenderer {
    /// Who a grouped slot holds: `(kind, persistentID)` for a named draw, `(kind, nth unnamed)` otherwise. Hashed by
    /// the kind's hash worked out once (`InstanceLayout`), not by its string.
    struct InstanceSlotKey: Hashable {
        let kind: MeshKind
        let kindHash: Int
        let id: UInt64
        let ordinal: Int32
        static func == (a: InstanceSlotKey, b: InstanceSlotKey) -> Bool {
            a.id == b.id && a.ordinal == b.ordinal && a.kindHash == b.kindHash && a.kind == b.kind
        }
        func hash(into hasher: inout Hasher) {
            hasher.combine(kindHash); hasher.combine(id); hasher.combine(ordinal)
        }
    }

    /// The keys of `instances` taken in `order` (source indices in grouped-slot order).
    static func slotKeys(_ instances: [InstanceRef], order: [Int]) -> [InstanceSlotKey] {
        slotKeys(instances, order: order, kindHashes: instances.map(\.meshKind.hashValue))
    }

    /// `slotKeys(_:order:)` with each source draw's kind hash already worked out (index-parallel to `instances`).
    static func slotKeys(_ instances: [InstanceRef], order: [Int], kindHashes: [Int]) -> [InstanceSlotKey] {
        var unnamed: [HashedKind: Int32] = [:]
        var keys: [InstanceSlotKey] = []
        keys.reserveCapacity(order.count)
        for i in order {
            let ref = instances[i]
            let h = kindHashes[i]
            if ref.persistentID != 0 {
                keys.append(InstanceSlotKey(kind: ref.meshKind, kindHash: h, id: ref.persistentID, ordinal: 0))
            } else {
                let k = HashedKind(ref.meshKind, hash: h)
                let n = unnamed[k, default: 0]
                unnamed[k] = n + 1
                keys.append(InstanceSlotKey(kind: ref.meshKind, kindHash: h, id: 0, ordinal: n))
            }
        }
        return keys
    }

    /// For each of this frame's slots, the slot its key held last frame, or nil (a draw new this frame).
    static func previousSlots(now: [InstanceSlotKey], before: [InstanceSlotKey]) -> [Int?] {
        let was = Dictionary(before.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        return now.map { was[$0] }
    }

    /// After `uploadInstances` has written this frame's grouped records: when the layout moved, lay last
    /// frame's records out like this frame's and bind them as the previous instances (`prevAlignedActive`).
    ///
    /// `keys` are this frame's slot keys (`InstanceLayout.slotKeys`). While the make-up holds they are the very array
    /// last frame stored, so the compare below is a storage-identity check, not a walk over every key.
    func alignPreviousInstances(keys: [InstanceSlotKey]) {
        // A host kernel may rewrite slots on the GPU after this upload, so the CPU's copy of last frame
        // would not be what was drawn: keep the plain ping-pong there (the pre-alignment behaviour).
        guard onEncodeGPUInstances == nil else { gbs.lastSlotKeys.removeAll(); return }
        defer { gbs.lastSlotKeys = keys }
        guard !gbs.lastSlotKeys.isEmpty, keys != gbs.lastSlotKeys,
              let ring = alignedRing(capacity: keys.count) else { return }
        let from = Self.previousSlots(now: keys, before: gbs.lastSlotKeys)
        let stride = MemoryLayout<IlluminatoramaInstance>.stride
        let raw = gbs.useBufferA ? gbs.instanceBufferB! : gbs.instanceBufferA!
        let prev = raw.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: raw.length / stride)
        let cur = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: keys.count)
        let dst = ring[lts.frameRingIndex].contents().bindMemory(to: IlluminatoramaInstance.self, capacity: keys.count)
        for (slot, old) in from.enumerated() { dst[slot] = old.map { prev[$0] } ?? cur[slot] }
        gbs.prevAlignedActive = true
    }

    /// The aligned ring, one buffer per frame in flight (a frame still on the GPU keeps reading its own),
    /// at least `capacity` records long.
    private func alignedRing(capacity: Int) -> [MTLBuffer]? {
        let slots = lts.frameUniformRing.count
        if gbs.prevAlignedRing.count == slots, gbs.prevAlignedCapacity >= capacity { return gbs.prevAlignedRing }
        let n = max(capacity, gbs.instanceCapacity)
        var ring: [MTLBuffer] = []
        for i in 0..<slots {
            guard let b = device.makeBuffer(length: MemoryLayout<IlluminatoramaInstance>.stride * n,
                                            options: .storageModeShared) else { return nil }
            b.label = "Illuminatorama.prevInstancesAligned.\(i)"
            ring.append(b)
        }
        gbs.prevAlignedRing = ring
        gbs.prevAlignedCapacity = n
        return ring
    }
}
