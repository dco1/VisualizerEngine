import Foundation

// ── The draw list's make-up, kept between frames ────────────────────────────
//
// `uploadInstances` groups the host's draws by mesh kind. Before this it did that from scratch every frame the list
// changed at all: two dictionary probes a draw (each hashing the kind's string — a `.custom` name is often 40+
// bytes), a slot key a draw for the motion-vector history (retaining the string again), the keys compared with last
// frame's, and `meshes[kind]` hashed again by every pass that draws a group. About 4 µs a draw on an iPhone 15 Pro
// for a list that had only MOVED: same kinds, same identities, same order (a turning gallery, a swinging rope).
//
// Now the make-up is kept (`InstanceLayout`): each draw's kind and identity, the grouping derived from them, the slot
// keys, and each group's mesh. A frame whose list holds that make-up (a string compare and an integer compare a draw,
// the string compare usually a pointer compare) reuses all of it and only copies the records. A frame whose make-up
// changed regroups, hashing each kind once (and not at all down a run of one kind) and keying the dictionaries by
// that hash (`HashedKind`).

extension IlluminatoramaRenderer {
    /// A mesh kind with its hash worked out once: a dictionary keyed by it hashes one integer, not the kind's string.
    struct HashedKind: Hashable {
        let kind: MeshKind
        let hash: Int
        init(_ kind: MeshKind, hash: Int) { self.kind = kind; self.hash = hash }
        static func == (a: HashedKind, b: HashedKind) -> Bool { a.hash == b.hash && a.kind == b.kind }
        func hash(into hasher: inout Hasher) { hasher.combine(hash) }
    }

    /// What `uploadInstances` derives from the draw list's make-up (each draw's kind and `persistentID`, in order).
    struct InstanceLayout {
        /// Each source draw's kind, identity and kind hash, in the host's order.
        var kinds: [MeshKind] = []
        var ids: [UInt64] = []
        var kindHashes: [Int] = []
        /// Grouped slot → source index.
        var order: [Int] = []
        var groups: [MeshDrawGroup] = []
        var groupRange: [MeshKind: Range<Int>] = [:]
        /// Who sits in each grouped slot (`alignPreviousInstances`).
        var slotKeys: [InstanceSlotKey] = []
        /// Bumped on every rebuild, so what's cached against a layout (`meshGroupMeshes`) knows when it's stale.
        var serial = 0

        /// True when `instances` has this make-up: same count, and draw by draw the same kind and identity.
        func holds(_ instances: [InstanceRef]) -> Bool {
            guard instances.count == ids.count else { return false }
            return instances.withUnsafeBufferPointer { b in
                kinds.withUnsafeBufferPointer { k in
                    ids.withUnsafeBufferPointer { d in
                        for i in 0..<b.count where b[i].persistentID != d[i] || b[i].meshKind != k[i] { return false }
                        return true
                    }
                }
            }
        }
    }

    /// The layout of `instances`, grouped by kind in first-seen order (so a stable list gives a stable buffer layout).
    /// A kind's hash is taken from `old` where the same source slot held the same kind, else worked out once per run of
    /// draws of one kind.
    static func layout(of instances: [InstanceRef], reusing old: InstanceLayout) -> InstanceLayout {
        var l = InstanceLayout()
        l.serial = old.serial &+ 1
        let n = instances.count
        l.kinds.reserveCapacity(n); l.ids.reserveCapacity(n); l.kindHashes.reserveCapacity(n)
        var groupOf: [HashedKind: Int] = [:]
        var members: [[Int]] = []
        var groupKinds: [MeshKind] = []
        var run: (kind: MeshKind, hash: Int, group: Int)?
        for i in 0..<n {
            let ref = instances[i]
            let kind = ref.meshKind
            let hash: Int, group: Int
            if let r = run, r.kind == kind {
                (hash, group) = (r.hash, r.group)
            } else {
                hash = i < old.kinds.count && old.kinds[i] == kind ? old.kindHashes[i] : kind.hashValue
                let key = HashedKind(kind, hash: hash)
                if let g = groupOf[key] {
                    group = g
                } else {
                    group = members.count
                    groupOf[key] = group
                    members.append([])
                    groupKinds.append(kind)
                }
                run = (kind, hash, group)
            }
            l.kinds.append(kind); l.ids.append(ref.persistentID); l.kindHashes.append(hash)
            members[group].append(i)
        }
        l.order.reserveCapacity(n)
        l.groups.reserveCapacity(members.count)
        l.groupRange.reserveCapacity(members.count)
        for (g, m) in members.enumerated() {
            let start = l.order.count
            l.order.append(contentsOf: m)
            l.groups.append(MeshDrawGroup(kind: groupKinds[g], start: start, count: m.count))
            l.groupRange[groupKinds[g]] = start ..< start + m.count
        }
        l.slotKeys = slotKeys(instances, order: l.order, kindHashes: l.kindHashes)
        return l
    }

    /// `gbs.meshGroupMeshes` for this frame's groups: each group's registered mesh, looked up once per regroup or per
    /// change to the mesh table (`setMesh`, `registerMesh`, …), not once per group per pass.
    func refreshMeshGroupMeshes() {
        let key = (gbs.instanceLayout.serial, kMisc.meshesVersion)
        guard gbs.meshGroupMeshesKey != key || gbs.meshGroupMeshes.count != gbs.meshGroups.count else { return }
        gbs.meshGroupMeshesKey = key
        gbs.meshGroupMeshes = gbs.meshGroups.map { meshes[$0.kind] }
    }

    /// Group `gi`'s mesh: the cached one (`refreshMeshGroupMeshes`), else the table's.
    @inline(__always)
    func groupMesh(_ gi: Int, _ group: MeshDrawGroup) -> IlluminatoramaMesh? {
        gi < gbs.meshGroupMeshes.count ? gbs.meshGroupMeshes[gi] : meshes[group.kind]
    }

    /// `a == b` for two draw lists, cheaper: storage identity first (the host kept its array), then per draw the
    /// identity, kind and CPU-only fields, and the 400-odd-byte GPU record by its bytes, falling back to its ~50
    /// field-by-field compares only when the bytes differ. Bitwise-equal records are equal (a NaN field that held still
    /// counts as unchanged, as it should); bytes that differ in padding alone still compare equal through `==`.
    static func sameInstances(_ a: [InstanceRef], _ b: [InstanceRef]) -> Bool {
        guard a.count == b.count else { return false }
        let dataOffset = MemoryLayout<InstanceRef>.offset(of: \InstanceRef.data)!
        let size = MemoryLayout<IlluminatoramaInstance>.size
        return a.withUnsafeBufferPointer { pa in
            b.withUnsafeBufferPointer { pb in
                guard pa.count > 0, let ba = pa.baseAddress, let bb = pb.baseAddress else { return true }
                if ba == bb { return true }
                for i in 0..<pa.count {
                    let x = ba + i, y = bb + i
                    guard x.pointee.persistentID == y.pointee.persistentID,
                          x.pointee.lightEmission == y.pointee.lightEmission,
                          x.pointee.superquadricShape == y.pointee.superquadricShape,
                          x.pointee.meshKind == y.pointee.meshKind else { return false }
                    if memcmp(UnsafeRawPointer(x) + dataOffset, UnsafeRawPointer(y) + dataOffset, size) != 0,
                       x.pointee.data != y.pointee.data { return false }
                }
                return true
            }
        }
    }
}
