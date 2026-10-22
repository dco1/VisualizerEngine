import simd

/// One cluster of the LOD DAG — flat, fixed-size, no pointers: the record S2 uploads and a compute
/// thread tests on its own.
///
/// **The cut rule** (`SignackSelection.isInCut`): draw a cluster when its OWN error projects under
/// the threshold and its PARENT group's error does not. Every cluster made from one group shares
/// that group's (`lodSphere`, `lodError`); every cluster simplified together in one group shares
/// (`parentSphere`, `parentError`); and up the DAG spheres only grow and errors only rise. So the
/// decision is identical across a group, and across a group boundary exactly one side is drawn —
/// which is why a mixed-level cut has neither cracks nor overlaps.
public struct SignackCluster: Sendable, Equatable {
    /// First INDEX (not triangle) of this cluster's run in `SignackHierarchy.indices`.
    public var indexStart: UInt32
    public var triangleCount: UInt32
    /// 0 = the source mesh.
    public var level: UInt16
    /// The group this cluster was simplified in (its parent group), or −1 for a root.
    public var group: Int32
    /// Bounds of the cluster's own triangles — for culling.
    public var bounds: SignackSphere
    public var cone: SignackNormalCone
    /// Where this cluster's error is measured from, and how large it is (object units). Level 0: 0.
    public var lodSphere: SignackSphere
    public var lodError: Float
    /// The same for the group that replaces it one level up; `parentError` is +∞ for a root.
    public var parentSphere: SignackSphere
    public var parentError: Float

    public var isRoot: Bool { !parentError.isFinite }
    public var indexRange: Range<Int> { Int(indexStart) ..< Int(indexStart) + 3 * Int(triangleCount) }
}
