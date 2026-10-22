import Foundation

/// Picks which pre-baked rung of a `SignackBudgetLadder` a frame draws, from how big the mesh is
/// on screen. Per-frame work is a handful of flops — never a rebake.
///
/// The rule: a rung is good enough when it carries at least `trianglesPerPixel` budgeted
/// triangles per pixel of the mesh's projected disc. Take the thinnest rung that is good enough.
/// Leaf cards overlap several deep, so the target is above 1 by design.
///
/// **Hysteresis.** A host passes the rung it drew last frame. Thinning (moving to a higher rung
/// index) needs `hysteresis` × the target of headroom; thickening happens as soon as the current
/// rung falls short. An orbit at constant distance therefore never flickers between rungs, and a
/// rung change — which on the ray-traced lanes is a TLAS topology change — happens once per real
/// change of distance.
public struct SignackLODSelector: Sendable, Equatable {
    public var trianglesPerPixel: Float = 1.5
    public var hysteresis: Float = 1.3

    public init(trianglesPerPixel: Float = 1.5, hysteresis: Float = 1.3) {
        self.trianglesPerPixel = trianglesPerPixel
        self.hysteresis = hysteresis
    }

    /// Projected radius in pixels of a sphere of `radius` at `distance`, for a perspective camera
    /// of vertical field of view `fovY` (radians) drawing `viewportHeight` pixels.
    public static func projectedRadius(radius: Float, distance: Float, fovY: Float,
                                       viewportHeight: Float) -> Float {
        let d = max(distance, radius * 0.5, 1e-3)
        return radius / (d * tan(fovY * 0.5)) * viewportHeight * 0.5
    }

    /// Projected radius in pixels for an orthographic camera of half-height `halfHeight` (m).
    public static func projectedRadius(radius: Float, orthographicHalfHeight halfHeight: Float,
                                       viewportHeight: Float) -> Float {
        radius / max(halfHeight, 1e-3) * viewportHeight * 0.5
    }

    /// The rung to draw. `rungTriangles` is each rung's budgeted triangle count (descending);
    /// `previous` is last frame's rung, or nil for a first frame.
    public func rung(rungTriangles: [Int], projectedRadiusPixels r: Float, previous: Int?) -> Int {
        guard rungTriangles.count > 1, r.isFinite else { return 0 }
        let target = trianglesPerPixel * Float.pi * r * r
        // Thinnest rung that meets the target (rung 0 if none does).
        var best = 0
        for i in rungTriangles.indices.reversed() where Float(rungTriangles[i]) >= target {
            best = i
            break
        }
        guard let prev = previous, prev >= 0, prev < rungTriangles.count else { return best }
        if best > prev {
            // Thinning: only once the thinner rung clears the target with headroom.
            var thin = prev
            for i in ((prev + 1)...best).reversed() where Float(rungTriangles[i]) >= target * hysteresis {
                thin = i
                break
            }
            return thin
        }
        return best   // thickening (or unchanged): immediately.
    }
}
