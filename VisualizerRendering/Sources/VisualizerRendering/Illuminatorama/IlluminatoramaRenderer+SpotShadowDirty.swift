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
    /// DH-1680 — at most this many boxes per swapped mesh (`IlluminatoramaMesh.shapeDiff(maxRuns:)`).
    static let swapDiffMaxRuns = 8

    /// The fields the shadow depth pass reads — the same set `updateSceneStability`'s shape check
    /// compares. Two instances with equal keys rasterise identical depth.
    private struct ShadowShapeKey: Hashable {
        let kind: MeshKind
        let m: SIMD16<Float>
        let swayMode: UInt32
        let lean: Float
        let jostle: Float
        let sinkDepth: Float, sinkKey: Float   // DH-1668 — `fieldSinkOffset` reads tagGlow.y/.z
        init(_ r: InstanceRef) {
            kind = r.meshKind
            let c = r.data.modelMatrix.columns
            m = SIMD16(lowHalf: SIMD8(lowHalf: c.0, highHalf: c.1), highHalf: SIMD8(lowHalf: c.2, highHalf: c.3))
            swayMode = UInt32(truncatingIfNeeded: Int(r.data.swayMode))
            lean = r.data.swayLean
            jostle = r.data.swayJostle
            sinkDepth = r.data.tagGlow.y
            sinkKey = r.data.tagGlow.z
        }
    }

    /// Record the casters whose shadow shape differs between `old` (last frame's upload) and `new`
    /// — into the spot atlas's record (DH-1590) and the sun cascades' (DH-1668), each kept only
    /// while its consumer's per-region reuse is on.
    func recordShadowCasterChange(old: [InstanceRef], new: [InstanceRef]) {
        shd.lastShapeChangeTally = (0, 0, 0, old.count == new.count)
        guard casterChangeRecording else { return }
        if old.count == new.count {
            // Same length: the multiset difference is contained in the differing indices (both sides).
            var diffMemo: [SwapPair: IlluminatoramaMesh.ShapeDiff] = [:]
            for i in 0..<new.count where ShadowShapeKey(old[i]) != ShadowShapeKey(new[i]) {
                if old[i].meshKind != new[i].meshKind {
                    shd.lastShapeChangeTally.kind += 1
                    // DH-1668 — a mesh SWAP under an otherwise identical instance: dirty only where
                    // the triangles differ, not the whole of both meshes (+ShapeDiff.swift).
                    if let regions = swapRegion(old[i], new[i], memo: &diffMemo) {
                        for b in regions { guard addCasterChangeBounds(b, kind: new[i].meshKind) else { return } }
                        continue
                    }
                } else if old[i].data.modelMatrix != new[i].data.modelMatrix { shd.lastShapeChangeTally.matrix += 1 }
                else { shd.lastShapeChangeTally.other += 1 }
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
            shd.lastShapeChangeTally.other += 1
            guard addCasterChange(e.ref) else { return }
        }
    }

    /// Is any consumer still recording bounded changes?
    private var casterChangeRecording: Bool {
        (spotShadowPerSliceDirtyEnabled && !shd.pendingCasterChangeUnbounded)
            || (sunCascadeReuseEnabled && !shd.cascadeChange.unbounded)
    }

    /// Add one instance's world bounds to every recording consumer; false (stop) once none is
    /// recording — the bound could not be computed (every record goes unbounded) or each record
    /// passed its cap.
    private func addCasterChange(_ r: InstanceRef) -> Bool {
        // DH-1668 — an OLD ref's mesh may already be gone from the table (a host that swapped it
        // out); `departedMeshes` still holds it for exactly this diff.
        guard let mesh = meshes[r.meshKind] ?? shd.departedMeshes[r.meshKind], mesh.cpuAuthoredVertices,
              !fxs.gpuRepackTasks.contains(where: { $0.kind == r.meshKind }),
              let local = mesh.boundingSphere,
              let s = IlluminatoramaCullBounds.worldSphere(local: local, instance: r.data,
                                                           treeWindStrength: treeWindStrength,
                                                           windMax: { mesh.windAttributeMax })
        else {
            markCasterChangeUnbounded(meshes[r.meshKind] == nil && shd.departedMeshes[r.meshKind] == nil
                                      ? "no mesh for \(r.meshKind)" : "unboundable mesh \(r.meshKind)")
            return false
        }
        var b = IlluminatoramaCullBounds.empty
        b.formUnion(center: s.center, radius: s.radius)
        return addCasterChangeBounds(b, kind: r.meshKind)
    }

    private struct SwapPair: Hashable { let a: ObjectIdentifier, b: ObjectIdentifier }

    /// For two refs that differ ONLY in their mesh (same matrix, no sway, no live wind, no field
    /// sink): `[]` when the meshes rasterise identical depth, else the world bounds of the triangles
    /// that differ — one box, or one per differing run (DH-1680, `swapDiffRunsEnabled`); `nil` = not
    /// a pure swap or not comparable (the caller then dirties both whole meshes).
    private func swapRegion(_ o: InstanceRef, _ n: InstanceRef,
                            memo: inout [SwapPair: IlluminatoramaMesh.ShapeDiff]) -> [IlluminatoramaCullBounds]? {
        let d = n.data
        guard swapDiffEnabled, o.data.modelMatrix == d.modelMatrix,
              o.data.swayMode == 0, d.swayMode == 0, o.data.swayJostle == d.swayJostle,
              treeWindStrength * max(o.data.windScale, d.windScale) == 0,
              o.data.tagGlow.y == d.tagGlow.y, o.data.tagGlow.z == d.tagGlow.z,
              d.tagGlow.y <= 0 || fieldSinkProgress <= 0,
              let ma = meshes[o.meshKind] ?? shd.departedMeshes[o.meshKind],
              let mb = meshes[n.meshKind] ?? shd.departedMeshes[n.meshKind],
              !fxs.gpuRepackTasks.contains(where: { $0.kind == o.meshKind || $0.kind == n.meshKind })
        else { return nil }
        let key = SwapPair(a: ObjectIdentifier(ma), b: ObjectIdentifier(mb))
        let diff = memo[key] ?? IlluminatoramaMesh.shapeDiff(ma, mb, maxRuns: swapDiffRunsEnabled ? Self.swapDiffMaxRuns : 1)
        memo[key] = diff
        switch diff {
        case .identical: return []
        case .whole: return nil
        case .region(let lo, let hi):
            guard let b = IlluminatoramaCullBounds.worldBox(lo: lo, hi: hi, model: d.modelMatrix) else { return nil }
            return [b]
        case .regions(let boxes):
            var out: [IlluminatoramaCullBounds] = []
            for r in boxes {
                guard let b = IlluminatoramaCullBounds.worldBox(lo: r.lo, hi: r.hi, model: d.modelMatrix) else { return nil }
                out.append(b)
            }
            return out
        }
    }

    /// Add one world bound to every recording consumer; false once none is recording.
    private func addCasterChangeBounds(_ b: IlluminatoramaCullBounds, kind: MeshKind? = nil) -> Bool {
        if spotShadowPerSliceDirtyEnabled && !shd.pendingCasterChangeUnbounded {
            if shd.pendingCasterChange.count < Self.casterChangeBoundsCap {
                shd.pendingCasterChange.append(b)
                shd.pendingCasterChangeKinds.append(kind)
            } else {
                shd.pendingCasterChange.removeAll(keepingCapacity: true)
                shd.pendingCasterChangeKinds.removeAll(keepingCapacity: true)
                shd.pendingCasterChangeUnbounded = true
                shd.casterChangeUnboundedReason = "more than \(Self.casterChangeBoundsCap) changed casters"
            }
        }
        if sunCascadeReuseEnabled {
            shd.cascadeChange.add(b, cap: Self.casterChangeBoundsCap)
            if shd.cascadeChange.unbounded {
                shd.casterChangeUnboundedReason = "more than \(Self.casterChangeBoundsCap) changed casters"
            }
        }
        return casterChangeRecording
    }

    /// A change nothing can bound (a mesh replaced under a live kind, an unreadable caster): every
    /// spot slice and every sun cascade re-renders on its next encode.
    func markCasterChangeUnbounded(_ reason: String) {
        shd.casterChangeUnboundedReason = reason
        shd.pendingCasterChange.removeAll(keepingCapacity: true)
        shd.pendingCasterChangeKinds.removeAll(keepingCapacity: true)
        shd.pendingCasterChangeUnbounded = true
        shd.cascadeChange.markUnbounded()
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

    /// The last shape change's differing instances by cause (mesh swapped, matrix moved, other —
    /// sway, sink, or the multiset path's insertions), and whether the instance count held. Counts
    /// stop where recording stopped (a record that went unbounded). Diagnostic.
    public var shadowLastShapeChangeTally: (kind: Int, matrix: Int, other: Int, sameCount: Bool) { shd.lastShapeChangeTally }

    /// DH-1673 — the mesh kind behind each `spotShadowLastCasterChange` box, parallel by index (nil
    /// = a box not owned by one kind). Diagnostic: lets a host name what moved.
    public var spotShadowLastCasterChangeKinds: [MeshKind?] { shd.lastCasterChangeKinds }

    /// DH-1673 — for each slice the last spot-atlas encode tested (its light matrix), the indices
    /// of the `spotShadowLastCasterChange` boxes its clip volume does NOT exclude: the changes that
    /// made it re-render. Diagnostic: which caster costs which slice.
    public var spotShadowLastSliceTouches: [[Int]] {
        shd.lastSpotShadowMats.map { m in
            let clip = IlluminatoramaClipVolume(m)
            return shd.lastCasterChange.indices.filter { !clip.excludes(shd.lastCasterChange[$0]) }
        }
    }

    /// DH-1673 — who owns each spot-atlas slice this frame, and where: "spot 3 (x, y, z)" /
    /// "area 0 (…)" (a window portal). Diagnostic.
    public var spotShadowSliceOwners: [String] {
        shd.spotShadowCasters.map { c in
            func f(_ p: SIMD3<Float>) -> String { String(format: "(%.1f, %.1f, %.1f)", p.x, p.y, p.z) }
            switch c.owner {
            case .spot(let i): return i < spotLights.count ? "spot \(i) \(f(spotLights[i].position))" : "spot \(i)"
            case .area(let i): return i < areaLights.count ? "area \(i) \(f(areaLights[i].center))" : "area \(i)"
            }
        }
    }

    /// CPU ms the last caster-change record took (the shape diff included). Diagnostic.
    public var shadowCasterRecordMs: Double { shd.lastCasterRecordMs }

    /// Why the caster-change record last went unbounded ("" = never). Diagnostic.
    public var shadowCasterChangeUnboundedReason: String { shd.casterChangeUnboundedReason }

    func consumeShadowCasterChanges() {
        shd.lastCasterChange = shd.pendingCasterChange
        shd.lastCasterChangeKinds = shd.pendingCasterChangeKinds
        shd.lastCasterChangeUnbounded = shd.pendingCasterChangeUnbounded
        shd.pendingCasterChange.removeAll(keepingCapacity: true)
        shd.pendingCasterChangeKinds.removeAll(keepingCapacity: true)
        shd.pendingCasterChangeUnbounded = false
    }
}
