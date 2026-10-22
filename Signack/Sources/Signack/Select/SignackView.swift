import Foundation
import simd

/// A camera, as LOD selection sees it: where it is (in the hierarchy's OBJECT space), how many
/// pixels one object unit covers, and how many pixels of error are acceptable.
///
/// Plain values, no references — the same struct S2 uploads as a constant buffer.
public struct SignackView: Sendable, Equatable {
    /// Camera position in object space.
    public var position: SIMD3<Float>
    /// Perspective: pixels covered by one object unit at distance one (viewportHeight / 2·tan(fovY/2)).
    /// Orthographic: pixels per object unit, flat (see `orthographic`).
    public var pixelScale: Float
    public var isOrthographic: Bool
    /// Largest acceptable projected error, in pixels.
    public var threshold: Float
    /// Optional object-space frustum for culling (`SignackSelection.select(_:view:cull:)`).
    public var frustum: SignackFrustum?
    /// Object units added to every cluster's bounding radius before the frustum test — what the
    /// host's vertex stage can move a vertex beyond its rest position (vegetation wind).
    public var cullPadding: Float = 0

    public init(position: SIMD3<Float>, pixelScale: Float, isOrthographic: Bool = false,
                threshold: Float = 1, frustum: SignackFrustum? = nil) {
        self.position = position
        self.pixelScale = pixelScale
        self.isOrthographic = isOrthographic
        self.threshold = threshold
        self.frustum = frustum
    }

    /// A perspective camera of vertical field of view `fovY` (radians) drawing `viewportHeight` px.
    public static func perspective(position: SIMD3<Float>, fovY: Float, viewportHeight: Float,
                                   threshold: Float = 1) -> SignackView {
        SignackView(position: position, pixelScale: viewportHeight / (2 * tan(fovY * 0.5)),
                    threshold: threshold)
    }

    /// An orthographic camera showing `halfHeight` object units above and below centre.
    public static func orthographic(halfHeight: Float, viewportHeight: Float,
                                    threshold: Float = 1) -> SignackView {
        SignackView(position: .zero, pixelScale: viewportHeight / (2 * max(halfHeight, 1e-6)),
                    isOrthographic: true, threshold: threshold)
    }

    /// A view-INDEPENDENT "camera" whose cut keeps every cluster with error ≤ `tolerance` object
    /// units — the fixed proxy cut a ray tracer can keep for good.
    public static func objectSpace(tolerance: Float) -> SignackView {
        SignackView(position: .zero, pixelScale: 1, isOrthographic: true, threshold: tolerance)
    }

    /// `error` (object units) measured at `sphere`, in pixels. Conservative: taken at the sphere's
    /// nearest point, and +∞ with the camera inside it. Monotone in the sphere (a containing sphere
    /// never projects smaller) and in the error — the two facts the crack-free cut rests on.
    @inlinable public func projectedError(_ sphere: SignackSphere, _ error: Float) -> Float {
        guard error > 0 else { return 0 }
        guard error.isFinite else { return .infinity }
        if isOrthographic { return error * pixelScale }
        let d = simd_distance(position, sphere.center) - sphere.w
        return d > 1e-4 ? error * pixelScale / d : .infinity
    }
}

/// Six object-space planes, inside where `dot(plane.xyz, p) + plane.w ≥ 0`.
public struct SignackFrustum: Sendable, Equatable {
    public var planes: [SIMD4<Float>]

    /// From a Metal-convention (clip z ∈ [0, 1]) projection × view (× model, for object space)
    /// matrix — Gribb–Hartmann plane extraction, normalised.
    public init(clip m: simd_float4x4) {
        func row(_ i: Int) -> SIMD4<Float> { SIMD4(m.columns.0[i], m.columns.1[i], m.columns.2[i], m.columns.3[i]) }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map { p in
            let l = simd_length(SIMD3(p.x, p.y, p.z))
            return l > 0 ? p / l : p
        }
    }

    /// Could any part of `sphere` be inside?
    @inlinable public func intersects(_ sphere: SignackSphere) -> Bool {
        for p in planes where simd_dot(SIMD3(p.x, p.y, p.z), sphere.center) + p.w < -sphere.w { return false }
        return true
    }
}
