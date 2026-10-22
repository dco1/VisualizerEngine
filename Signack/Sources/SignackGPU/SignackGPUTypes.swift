import simd
import Signack

/// One cluster as the GPU reads it — the byte-for-byte mirror of `SignackGPUCluster` in
/// `SignackCut.metal` (64 B; the shader static_asserts the size, `SignackGPUTests` the Swift side).
///
/// The CPU `SignackCluster` minus what the kernel never reads (level, group, normal cone), with a
/// root's +∞ parent error stored as the finite `rootError` sentinel (fast-math may fold +∞ away).
public struct SignackGPUCluster: Sendable, Equatable {
    public var bounds: SIMD4<Float>
    public var lodSphere: SIMD4<Float>
    public var parentSphere: SIMD4<Float>
    public var indexStart: UInt32
    public var triangleCount: UInt32
    public var lodError: Float
    public var parentError: Float

    /// ≥ this is "+∞" on the GPU (`SIGNACK_ROOT_ERROR`).
    public static let rootError: Float = 1.0e30

    public init(_ c: SignackCluster) {
        bounds = c.bounds
        lodSphere = c.lodSphere
        parentSphere = c.parentSphere
        indexStart = c.indexStart
        triangleCount = c.triangleCount
        lodError = c.lodError.isFinite ? min(c.lodError, Self.rootError) : Self.rootError
        parentError = c.parentError.isFinite ? min(c.parentError, Self.rootError) : Self.rootError
    }
}

/// The per-dispatch constants — mirror of `SignackGPUCutUniforms` in `SignackCut.metal` (144 B).
/// Built from the same `SignackView` the CPU reference selects with, so the two cannot drift.
public struct SignackGPUCutUniforms: Sendable, Equatable {
    public var plane0, plane1, plane2, plane3, plane4, plane5: SIMD4<Float>
    public var camera: SIMD4<Float>
    public var threshold: Float
    public var boundsPad: Float
    public var isOrthographic: UInt32
    public var clusterCount: UInt32
    public var cull: UInt32
    public var capacity: UInt32
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0

    /// A plane every sphere is inside (used when the view carries no frustum).
    static let passPlane = SIMD4<Float>(0, 0, 0, 1e30)

    /// The constants for selecting `view`'s cut of a hierarchy with `clusterCount` clusters into an
    /// output of `capacity` indices. With a frustum on the view, clusters outside it (by their
    /// bounds + `view.cullPadding`) are drawn only by the shadow passes.
    public init(view: SignackView, clusterCount: Int, capacity: Int) {
        let p = view.frustum?.planes ?? []
        func plane(_ i: Int) -> SIMD4<Float> { i < p.count ? p[i] : Self.passPlane }
        plane0 = plane(0); plane1 = plane(1); plane2 = plane(2)
        plane3 = plane(3); plane4 = plane(4); plane5 = plane(5)
        camera = SIMD4(view.position, view.pixelScale)
        threshold = view.threshold
        boundsPad = view.cullPadding
        isOrthographic = view.isOrthographic ? 1 : 0
        self.clusterCount = UInt32(clusterCount)
        cull = view.frustum == nil ? 0 : 1
        self.capacity = UInt32(capacity)
    }
}

/// Where the two indirect argument records sit in a cut's argument buffer.
public enum SignackGPUDrawArgs {
    /// The on-screen part of the cut — the camera's G-buffer draw.
    public static let primaryOffset = 0
    /// The whole cut, off-screen clusters included — every shadow pass (their casters reach past
    /// the camera frustum).
    public static let shadowOffset = 32
    public static let bufferLength = 64
}
