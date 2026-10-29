import simd

// PER-SLICE CASTER DIRTYING (Daydream DH-1590)
// ============================================
//
// A spot/area atlas slice's map depends on exactly two things: its light matrix and the casters
// inside that matrix's clip volume. DH-1064 made the matrix half per-slice (sticky slices, reuse
// decided against the matrix each slice was last rendered with), but the caster half stayed
// whole-scene: reuse was gated on `instanceShapeStableFrames >= 1`, so dragging ONE sofa — or a
// pendant's decaying swing changing its `swayLean` every frame (DH-1589) — re-rasterised every
// slice of the atlas, including cones on the far side of the house that cannot see it.
//
// So when the shape check misses, `updateSceneStability` records the world bounds of every
// instance whose shadow SHAPE differs between last frame and this one — BOTH poses: the new one
// (where the shadow now falls) and the old one (where it must be cleared from maps that used to
// hold it — the classic stale-shadow bug on the frame a caster LEAVES a cone). A slice whose clip
// volume excludes every recorded bound still holds a correct map and is reused.
//
// The argument is the visibility cull's (DH-0534): the bounds are `IlluminatoramaCullBounds
// .worldSphere`, which covers everything the shadow vertex stage can do to a vertex (wind, sway at
// ANY angle about its pivot, jostle), and a convex set outside one clip half-space produces no
// fragment — so an excluded change cannot have written, or un-written, one texel of that slice.
// The diff is a set difference of shape keys, so it is exact up to the bounds, never "roughly".
// Anything that cannot be bounded — a mesh with no CPU-readable vertices, a GPU-repacked kind, a
// degenerate matrix, more than `casterChangeBoundsCap` changes since the last encode — marks the
// change unbounded and every slice re-renders, the pre-DH-1590 behaviour.
//
// Changes ACCUMULATE until the atlas is encoded (`consumeShadowCasterChanges`), so a frame that
// rendered without encoding spot shadows cannot lose a move.
extension IlluminatoramaRenderer {

    /// Past this many changed casters since the last encode, the change is treated as unbounded:
    /// the per-slice test would cost more than it saves, and a mass change hits most slices anyway.
    static let casterChangeBoundsCap = 256

    /// The fields the shadow depth pass reads — the same set `updateSceneStability`'s shape check
    /// compares. Two instances with equal keys rasterise identical depth.
    private struct ShadowShapeKey: Hashable {
        let kind: MeshKind
        let m: SIMD16<Float>
        let swayMode: UInt32
        let lean: Float
        let jostle: Float
        init(_ r: InstanceRef) {
            kind = r.meshKind
            let c = r.data.modelMatrix.columns
            m = SIMD16(lowHalf: SIMD8(lowHalf: c.0, highHalf: c.1), highHalf: SIMD8(lowHalf: c.2, highHalf: c.3))
            swayMode = UInt32(truncatingIfNeeded: Int(r.data.swayMode))
            lean = r.data.swayLean
            jostle = r.data.swayJostle
        }
    }

    /// Record the casters whose shadow shape differs between `old` (last frame's upload) and `new`.
    func recordShadowCasterChange(old: [InstanceRef], new: [InstanceRef]) {
        guard spotShadowPerSliceDirtyEnabled, !shd.pendingCasterChangeUnbounded else { return }
        if old.count == new.count {
            // Same length: the multiset difference is contained in the differing indices (both sides).
            for i in 0..<new.count where ShadowShapeKey(old[i]) != ShadowShapeKey(new[i]) {
                guard addCasterChange(old[i]), addCasterChange(new[i]) else { return }
            }
            return
        }
        // Length changed (a piece added or removed — later slots shift): an exact multiset diff,
        // so an insertion does not mark every later instance as moved.
        var tally: [ShadowShapeKey: (count: Int, ref: InstanceRef)] = [:]
        tally.reserveCapacity(max(old.count, new.count))
        for r in old { tally[ShadowShapeKey(r), default: (0, r)].count += 1 }
        for r in new { tally[ShadowShapeKey(r), default: (0, r)].count -= 1 }
        for (_, e) in tally where e.count != 0 {
            guard addCasterChange(e.ref) else { return }
        }
    }

    /// Add one instance's world bounds; false (and the whole change marked unbounded) when it
    /// cannot be bounded or the cap is exceeded.
    private func addCasterChange(_ r: InstanceRef) -> Bool {
        guard shd.pendingCasterChange.count < Self.casterChangeBoundsCap,
              let mesh = meshes[r.meshKind], mesh.cpuAuthoredVertices,
              !fxs.gpuRepackTasks.contains(where: { $0.kind == r.meshKind }),
              let local = mesh.boundingSphere,
              let s = IlluminatoramaCullBounds.worldSphere(local: local, instance: r.data,
                                                           treeWindStrength: treeWindStrength,
                                                           windMax: { mesh.windAttributeMax })
        else {
            shd.pendingCasterChange.removeAll(keepingCapacity: true)
            shd.pendingCasterChangeUnbounded = true
            return false
        }
        var b = IlluminatoramaCullBounds.empty
        b.formUnion(center: s.center, radius: s.radius)
        shd.pendingCasterChange.append(b)
        return true
    }

    /// Could any recorded caster change have touched the map rasterised with `matrix`?
    func casterChangeTouches(_ matrix: simd_float4x4) -> Bool {
        if shd.pendingCasterChangeUnbounded { return true }
        if shd.pendingCasterChange.isEmpty { return false }
        let clip = IlluminatoramaClipVolume(matrix)
        return shd.pendingCasterChange.contains { !clip.excludes($0) }
    }

    /// The caster-change bounds the LAST spot-atlas encode tested its slices against (world AABBs,
    /// old and new poses), or nil when that change was unbounded (every slice re-rendered). Empty
    /// = no caster changed. Diagnostic: lets a gate see what the engine believed moved.
    public var spotShadowLastCasterChange: [(lo: SIMD3<Float>, hi: SIMD3<Float>)]? {
        shd.lastCasterChangeUnbounded ? nil : shd.lastCasterChange.map { ($0.lo, $0.hi) }
    }

    func consumeShadowCasterChanges() {
        shd.lastCasterChange = shd.pendingCasterChange
        shd.lastCasterChangeUnbounded = shd.pendingCasterChangeUnbounded
        shd.pendingCasterChange.removeAll(keepingCapacity: true)
        shd.pendingCasterChangeUnbounded = false
    }
}
