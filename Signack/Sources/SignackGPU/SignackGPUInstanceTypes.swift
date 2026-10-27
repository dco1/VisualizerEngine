import Metal
import simd
import Signack

/// One instance's WORLD bounds for the batched pass — mirror of `SignackInstanceBounds` (32 B).
public struct SignackGPUInstanceBounds: Sendable, Equatable {
    /// Centre + radius (include whatever the vertex stage can move a vertex — wind reach).
    public var sphere: SIMD4<Float>
    /// Object → world scale: the rungs' errors are object-space.
    public var scale: Float
    var pad0: Float = 0, pad1: Float = 0, pad2: Float = 0

    public init(center: SIMD3<Float>, radius: Float, scale: Float) {
        sphere = SIMD4(center, radius); self.scale = scale
    }
}

/// The camera as the batched pass sees it — WORLD space (each instance carries its own transform).
public struct SignackInstanceView: Sendable, Equatable {
    public var position: SIMD3<Float>
    /// Perspective: px per world unit at distance 1 (viewportHeight / 2·tan(fovY/2)). Ortho: px per unit.
    public var pixelScale: Float
    public var isOrthographic: Bool
    /// Acceptable projected error (px) when picking a rung.
    public var threshold: Float
    /// A plant whose bounding sphere projects narrower than this (px) draws as its impostor.
    public var impostorPixels: Float
    /// World-space frustum; nil = no culling (every instance on screen).
    public var frustum: SignackFrustum?
    /// Beyond this a plant is not drawn at all (m; 0 = no limit).
    public var maxDistance: Float

    public init(position: SIMD3<Float>, pixelScale: Float, isOrthographic: Bool = false, threshold: Float = 1,
                impostorPixels: Float = 0, frustum: SignackFrustum? = nil, maxDistance: Float = 0) {
        self.position = position; self.pixelScale = pixelScale; self.isOrthographic = isOrthographic
        self.threshold = threshold; self.impostorPixels = impostorPixels; self.frustum = frustum
        self.maxDistance = maxDistance
    }
}

/// Mirror of `SignackInstanceUniforms` in `SignackInstances.metal` (256 B).
struct SignackGPUInstanceUniforms {
    var plane0, plane1, plane2, plane3, plane4, plane5: SIMD4<Float>
    var camera: SIMD4<Float>
    var rungError0, rungError1: SIMD4<Float>
    var rungStart0, rungStart1: SIMD4<UInt32>
    var rungCount0, rungCount1: SIMD4<UInt32>
    var threshold: Float
    var impostorPixels: Float
    var rungCount: UInt32
    var impostorIndexCount: UInt32
    var instanceCount: UInt32
    var instanceStride: UInt32
    var isOrthographic: UInt32
    var cull: UInt32
    var maxDistance: Float
    var viewCount: UInt32 = 0          // DH-1493
    var impostorIndexStart: UInt32 = 0 // DH-1470
    var pad2: UInt32 = 0

    init(view: SignackInstanceView, batch: SignackGPUInstanceBatch) {
        let p = view.frustum?.planes ?? []
        func plane(_ i: Int) -> SIMD4<Float> { i < p.count ? p[i] : SIMD4(0, 0, 0, 1e30) }
        plane0 = plane(0); plane1 = plane(1); plane2 = plane(2); plane3 = plane(3); plane4 = plane(4); plane5 = plane(5)
        camera = SIMD4(view.position, view.pixelScale)
        let r = batch.rungs
        func err(_ i: Int) -> Float { i < r.count ? r[i].error : 0 }
        func start(_ i: Int) -> UInt32 { i < r.count ? UInt32(r[i].indexStart) : 0 }
        func count(_ i: Int) -> UInt32 { i < r.count ? UInt32(r[i].indexCount) : 0 }
        rungError0 = SIMD4(err(0), err(1), err(2), err(3)); rungError1 = SIMD4(err(4), err(5), err(6), err(7))
        rungStart0 = SIMD4(start(0), start(1), start(2), start(3)); rungStart1 = SIMD4(start(4), start(5), start(6), start(7))
        rungCount0 = SIMD4(count(0), count(1), count(2), count(3)); rungCount1 = SIMD4(count(4), count(5), count(6), count(7))
        threshold = view.threshold
        impostorPixels = view.impostorPixels
        rungCount = UInt32(r.count)
        impostorIndexCount = UInt32(batch.impostorIndexCount)
        impostorIndexStart = UInt32(batch.impostorIndexStart)
        instanceCount = UInt32(batch.instanceCount)
        instanceStride = UInt32(batch.instanceStride)
        isOrthographic = view.isOrthographic ? 1 : 0
        cull = view.frustum == nil ? 0 : 1
        maxDistance = view.maxDistance
    }
}

/// DH-1493 — the shadow views a planting batch is culled against: mirror of `SignackInstanceShadowViews`
/// (768 B, view v's six world-space planes at [6v, 6v+6)). Unused slots pass everything.
struct SignackGPUInstanceShadowViews {
    static let byteLength = 6 * SignackShadowViews.maxViews * 16
    var words: [SIMD4<Float>]

    init(_ views: [SignackFrustum]) {
        words = Array(repeating: SIMD4<Float>(0, 0, 0, 1e30), count: 6 * SignackShadowViews.maxViews)
        for (v, f) in views.prefix(SignackShadowViews.maxViews).enumerated() {
            for (p, plane) in f.planes.prefix(6).enumerated() { words[6 * v + p] = plane }
        }
    }

    func bind(_ enc: MTLComputeCommandEncoder, index: Int) {
        words.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: Self.byteLength, index: index) }
    }
}

extension SignackGPUInstanceBatch {
    /// DH-1493 — the CPU REFERENCE of the per-view membership: may this instance land in `view`?
    /// (The same sphere-vs-planes test the kernel makes; the parity test holds it to it.)
    public static func referenceInView(_ b: SignackGPUInstanceBounds, _ view: SignackFrustum) -> Bool {
        let c = SIMD3(b.sphere.x, b.sphere.y, b.sphere.z)
        return view.planes.allSatisfy { simd_dot(SIMD3($0.x, $0.y, $0.z), c) + $0.w >= -b.sphere.w }
    }

    /// The CPU REFERENCE of `signack_instances_classify` — the bucket an instance lands in and
    /// whether it is on screen, or nil beyond `maxDistance`. The parity test holds the kernel to it;
    /// a host's debug view colours by it (the same answer the GPU drew).
    public static func referenceBucket(_ b: SignackGPUInstanceBounds, view v: SignackInstanceView,
                                       rungs: [SignackRungs.Rung], impostor: Bool) -> (bucket: Int, onScreen: Bool)? {
        let c = SIMD3(b.sphere.x, b.sphere.y, b.sphere.z), r = b.sphere.w
        let dist = v.isOrthographic ? 1 : max(simd_distance(v.position, c) - r, 1e-3)
        if v.maxDistance > 0, dist > v.maxDistance { return nil }
        let px = v.pixelScale / dist
        var bucket = 0
        if impostor, 2 * r * px < v.impostorPixels { bucket = rungs.count } else {
            for k in 1..<max(1, rungs.count) where rungs[k].error * b.scale * px <= v.threshold { bucket = k }
        }
        let on = v.frustum.map { f in f.planes.allSatisfy { simd_dot(SIMD3($0.x, $0.y, $0.z), c) + $0.w >= -r } } ?? true
        return (bucket, on)
    }
}
