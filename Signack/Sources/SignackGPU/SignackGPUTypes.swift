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
    /// S4 — projected triangle area (px²) below which an on-screen triangle goes to the software
    /// rasterizer (`SignackGPURouting.maxArea`); 0 = none.
    public var softArea: Float
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
        softArea = 0
    }
}

/// Where the indirect argument records sit in a cut's argument buffer.
///
/// The output index buffer (capacity C) is laid out `[off-screen | on-screen hardware | … | soft]`:
/// shadow = `[0, S + P)`, primary = `[S, S + P)`, soft = `[C − Q, C)` (written from the top down,
/// because a triangle's route is only known when its cluster is copied).
public enum SignackGPUDrawArgs {
    /// The on-screen triangles the HARDWARE rasterizer draws — the camera's G-buffer draw. With no
    /// soft routing that is every on-screen cluster, exactly S2.
    public static let primaryOffset = 0
    /// The whole cut except the soft triangles — every shadow pass draws it AND the soft record
    /// (casters reach past the camera frustum).
    public static let shadowOffset = 32
    /// S4 — the on-screen triangles routed to the software rasterizer, as an indexed-indirect record
    /// over the same buffer. Shadow passes always draw it; a camera pass that does not rasterise it
    /// in software draws it with the hardware, so a routed triangle is never lost.
    public static let softOffset = 64
    /// S4 — `MTLDispatchThreadgroupsIndirectArguments` for ONE THREAD PER SOFT TRIANGLE at
    /// `softThreadsPerGroup` threads per threadgroup (the host's vertex stage for the soft range).
    public static let softDispatchOffset = 96
    public static let softThreadsPerGroup = 64
    public static let bufferLength = 128
}
