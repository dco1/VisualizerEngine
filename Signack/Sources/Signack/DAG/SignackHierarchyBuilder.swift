import simd

/// Bakes a `SignackHierarchy` — Nanite's offline half — from any static indexed triangle mesh.
///
///     level 0   cluster the source mesh (≤ 128 triangles each)
///     repeat    group neighbouring clusters (≈ 4) → merge each group → simplify it to half with
///               its OUTER boundary locked → split the result back into clusters (level + 1)
///     until     one cluster is left, or no group can get coarser (those become roots)
///
/// Each group's error is max(its simplification error, its children's errors) and its sphere
/// encloses its children's, so error rises monotonically up the DAG — the property the per-cluster
/// cut test needs to stay crack-free (see `SignackCluster`).
///
/// The kind of geometry lives entirely in the `SignackSimplifier` passed in; clustering, grouping,
/// locking and bookkeeping are shared. Pure and deterministic — groups simplify in parallel, but
/// results are merged in group order, so the same input bakes the same DAG on every run.
public struct SignackHierarchyBuilder: Sendable {
    public var clusterizer = SignackClusterizer()
    public var groupSize = 4
    public var maxLevels = 24
    /// A group that cannot get below this fraction of its triangles stops (its clusters become
    /// roots) — a level that saves almost nothing costs a level of index memory for nothing.
    public var minimumReduction: Float = 0.85

    public init(maxClusterTriangles: Int = 128, groupSize: Int = 4) {
        clusterizer = SignackClusterizer(maxTriangles: maxClusterTriangles)
        self.groupSize = max(2, groupSize)
    }

    /// Build the DAG of `indices` over `positions` (the host's vertex buffer, in order), with
    /// optional per-triangle `atoms` (indivisible units — leaf cards).
    public func build(positions: [SIMD3<Float>], indices: [UInt32], atoms: [UInt32]? = nil,
                      simplifier: some SignackSimplifier) -> SignackHierarchy {
        precondition(indices.count % 3 == 0, "indices must be triangles")
        precondition(atoms == nil || atoms!.count == indices.count / 3, "one atom per triangle")
        var mesh = SignackMeshView(positions: positions)
        let tris = stride(from: 0, to: indices.count, by: 3).map {
            SIMD3(indices[$0], indices[$0 + 1], indices[$0 + 2])
        }
        var all: [Work] = []
        for part in clusterizer.clusters(tris, atoms: atoms, mesh: mesh) {
            let t = part.map { tris[$0] }
            let a = atoms.map { at in part.map { at[$0] } }
            var area: Float = 0
            for x in t { area += Self.area(x, mesh) }
            all.append(Work(tris: t, atoms: a, level: 0, lodSphere: Self.sphere(t, mesh), lodError: 0,
                            sourceTriangles: t.count, sourceAtoms: a.map { Set($0).count } ?? t.count,
                            sourceArea: area))
        }
        var current = Array(all.indices)
        var retired = Set<UInt32>()
        var derived: [SignackDerivedVertex] = []
        var derivedIndex: [SignackDerivedVertex: UInt32] = [:]
        var groupCounter: Int32 = 0
        var level = 0
        let grouping = SignackGrouping(groupSize: groupSize)

        while current.count > 1, level < maxLevels {
            let groups = grouping.groups(current.map { all[$0].tris }, mesh: mesh)
            // A position touched by two groups (or by a retired root) is a group boundary: locked.
            var owner: [UInt32: Int] = [:]
            var shared = retired
            for (gi, g) in groups.enumerated() {
                for c in g { for t in all[current[c]].tris { for k in 0..<3 {
                    let p = mesh.positionID[Int(t[k])]
                    if let o = owner[p] { if o != gi { shared.insert(p) } } else { owner[p] = gi }
                } } }
            }
            let inputs: [SignackGroupInput] = groups.map { g in
                let members = g.map { all[current[$0]] }
                let t = members.flatMap(\.tris)
                var locked = Set<UInt32>()
                for x in t { for k in 0..<3 {
                    let p = mesh.positionID[Int(x[k])]
                    if shared.contains(p) { locked.insert(p) }
                } }
                return SignackGroupInput(
                    triangles: t, atoms: atoms == nil ? nil : members.flatMap { $0.atoms ?? [] },
                    lockedPositions: locked, targetTriangles: t.count / 2,
                    sourceTriangles: members.reduce(0) { $0 + $1.sourceTriangles },
                    sourceAtoms: members.reduce(0) { $0 + $1.sourceAtoms },
                    sourceArea: members.reduce(0) { $0 + $1.sourceArea })
            }
            let snapshot = mesh
            let outputs = SignackParallel.map(groups.count) { simplifier.simplify(inputs[$0], mesh: snapshot) }

            var next: [Int] = []
            for (gi, g) in groups.enumerated() {
                let members = g.map { current[$0] }
                guard let out = outputs[gi], !out.triangles.isEmpty,
                      Float(out.triangles.count) <= minimumReduction * Float(inputs[gi].triangles.count) else {
                    // Roots: their geometry is final, so its boundary is frozen for every later level.
                    for m in members { for t in all[m].tris { for k in 0..<3 { retired.insert(mesh.positionID[Int(t[k])]) } } }
                    continue
                }
                let sphere = SignackBounds.union(members.map { all[$0].lodSphere })
                let error = members.reduce(out.error) { max($0, all[$1].lodError) }
                for m in members {
                    all[m].parentSphere = sphere; all[m].parentError = error; all[m].group = groupCounter
                }
                groupCounter += 1
                let ids: [UInt32] = out.newVertices.map { d in
                    if let v = derivedIndex[d] { return v }
                    let v = mesh.appendDerived(d)
                    derivedIndex[d] = v; derived.append(d)
                    return v
                }
                let flag = SignackGroupOutput.newVertexFlag
                let simplified = out.triangles.map { t in
                    SIMD3<UInt32>((0..<3).map { k in t[k] & flag != 0 ? ids[Int(t[k] & ~flag)] : t[k] })
                }
                let parts = clusterizer.clusters(simplified, atoms: out.atoms, mesh: mesh)
                let input = inputs[gi]
                var left = (input.sourceTriangles, input.sourceAtoms, input.sourceArea)
                for (k, part) in parts.enumerated() {
                    let t = part.map { simplified[$0] }
                    let share = Float(t.count) / Float(simplified.count)
                    let last = k == parts.count - 1
                    let st = last ? left.0 : Int((Float(input.sourceTriangles) * share).rounded())
                    let sa = last ? left.1 : Int((Float(input.sourceAtoms) * share).rounded())
                    let sr = last ? left.2 : input.sourceArea * share
                    left = (left.0 - st, left.1 - sa, left.2 - sr)
                    next.append(all.count)
                    all.append(Work(tris: t, atoms: out.atoms.map { at in part.map { at[$0] } },
                                    level: level + 1, lodSphere: sphere, lodError: error,
                                    sourceTriangles: max(st, 1), sourceAtoms: max(sa, 1), sourceArea: max(sr, 0)))
                }
            }
            if next.isEmpty { break }
            current = next
            level += 1
        }
        return Self.flatten(all, mesh: mesh, sourceVertexCount: positions.count, derived: derived)
    }

    /// A cluster while the DAG is being built.
    struct Work {
        var tris: [SIMD3<UInt32>]
        var atoms: [UInt32]?
        var level: Int
        var lodSphere: SignackSphere
        var lodError: Float
        var parentSphere = SignackSphere.zero
        var parentError = Float.infinity
        var group: Int32 = -1
        var sourceTriangles: Int
        var sourceAtoms: Int
        var sourceArea: Float
    }

    static func flatten(_ all: [Work], mesh: SignackMeshView, sourceVertexCount: Int,
                        derived: [SignackDerivedVertex]) -> SignackHierarchy {
        var indices: [UInt32] = []
        indices.reserveCapacity(all.reduce(0) { $0 + 3 * $1.tris.count })
        var clusters: [SignackCluster] = []
        clusters.reserveCapacity(all.count)
        for w in all {
            let start = UInt32(indices.count)
            var normals: [SIMD3<Float>] = []
            for t in w.tris {
                indices.append(t.x); indices.append(t.y); indices.append(t.z)
                let a = mesh.positions[Int(t.x)], b = mesh.positions[Int(t.y)], c = mesh.positions[Int(t.z)]
                let n = simd_cross(b - a, c - a)
                let l = simd_length(n)
                if l > 1e-20 { normals.append(n / l) }
            }
            clusters.append(SignackCluster(
                indexStart: start, triangleCount: UInt32(w.tris.count), level: UInt16(w.level),
                group: w.group, bounds: sphere(w.tris, mesh), cone: SignackNormalCone(normals: normals),
                lodSphere: w.lodSphere, lodError: w.lodError,
                parentSphere: w.parentSphere, parentError: w.parentError))
        }
        return SignackHierarchy(sourceVertexCount: sourceVertexCount, derivedVertices: derived,
                                indices: indices, clusters: clusters,
                                levelCount: (all.map(\.level).max() ?? 0) + 1,
                                positionID: mesh.positionID)
    }

    static func sphere(_ tris: [SIMD3<UInt32>], _ mesh: SignackMeshView) -> SignackSphere {
        var pts: [SIMD3<Float>] = []
        pts.reserveCapacity(tris.count * 3)
        for t in tris { for k in 0..<3 { pts.append(mesh.positions[Int(t[k])]) } }
        return SignackBounds.sphere(pts)
    }

    static func area(_ t: SIMD3<UInt32>, _ mesh: SignackMeshView) -> Float {
        let a = mesh.positions[Int(t.x)], b = mesh.positions[Int(t.y)], c = mesh.positions[Int(t.z)]
        return 0.5 * simd_length(simd_cross(b - a, c - a))
    }
}
