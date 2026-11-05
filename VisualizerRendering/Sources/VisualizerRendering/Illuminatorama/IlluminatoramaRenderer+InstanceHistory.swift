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
// pass over the records); a steady list costs a key compare.

extension IlluminatoramaRenderer {
    /// Who a grouped slot holds: `(kind, persistentID)` for a named draw, `(kind, nth unnamed)` otherwise.
    struct InstanceSlotKey: Hashable {
        let kind: MeshKind
        let id: UInt64
        let ordinal: Int32
    }

    /// The keys of `instances` taken in `order` (source indices in grouped-slot order).
    static func slotKeys(_ instances: [InstanceRef], order: [Int]) -> [InstanceSlotKey] {
        var unnamed: [MeshKind: Int32] = [:]
        var keys: [InstanceSlotKey] = []
        keys.reserveCapacity(order.count)
        for i in order {
            let ref = instances[i]
            if ref.persistentID != 0 {
                keys.append(InstanceSlotKey(kind: ref.meshKind, id: ref.persistentID, ordinal: 0))
            } else {
                let n = unnamed[ref.meshKind, default: 0]
                unnamed[ref.meshKind] = n + 1
                keys.append(InstanceSlotKey(kind: ref.meshKind, id: 0, ordinal: n))
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
    func alignPreviousInstances(order: [Int]) {
        // A host kernel may rewrite slots on the GPU after this upload, so the CPU's copy of last frame
        // would not be what was drawn: keep the plain ping-pong there (the pre-alignment behaviour).
        guard onEncodeGPUInstances == nil else { gbs.lastSlotKeys.removeAll(); return }
        let keys = Self.slotKeys(instances, order: order)
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
