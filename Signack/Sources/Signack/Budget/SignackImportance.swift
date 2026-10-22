import Foundation
import simd

/// How much one card is worth keeping, from what a camera outside the crown — and the sun above
/// it — can actually see. Camera-INDEPENDENT on purpose: the weight is baked once per mesh, and a
/// host picks among budgets per frame by distance (`SignackLODSelector`), so no ranking ever
/// depends on where the camera happens to be.
///
///     weight = (floor + (1 − floor) · shell^shellPower)       outer shell ≫ buried interior
///            × (1 − skyWeight + 2·skyWeight · elevation)       upper, sunlit crown ≫ underside
///            × (1 − facingWeight + facingWeight · |n · r̂|)     faces out of the crown ≫ edge-on
///            × clamp(areaPerTri / median, costMin, costMax)    coverage per triangle spent
///
/// `floor` keeps the interior sampled (sparser, never empty) so a crown seen through a gap is not
/// hollow — the lesson of the Visualizer Forest's importance LOD (`ForestController
/// .buildRTGeometry`, the `baseFloor` term), lifted here.
public struct SignackImportance: Sendable, Equatable {
    public var floor: Float = 0.12
    public var shellPower: Float = 3
    public var skyWeight: Float = 0.25
    public var facingWeight: Float = 0.3
    public var costMin: Float = 0.3
    public var costMax: Float = 1.6

    public init() {}

    /// The default leaf policy.
    public static let foliage = SignackImportance()

    /// Every card weighted 1 — the even-thinning control arm, for A/B.
    public static let uniform: SignackImportance = {
        var p = SignackImportance()
        p.floor = 1; p.skyWeight = 0; p.facingWeight = 0; p.costMin = 1; p.costMax = 1
        return p
    }()

    /// One weight per card, all finite and > 0.
    public func weights(for cards: [SignackCard]) -> [Float] {
        guard !cards.isEmpty else { return [] }
        let shell = SignackCrownShell(cards: cards)
        let median = Self.medianAreaPerTriangle(cards)
        return cards.map { c in
            let depth = shell.shellDepth(c.centroid)
            let shellTerm = floor + (1 - floor) * pow(depth, shellPower)
            let sky = 1 - skyWeight + 2 * skyWeight * shell.elevation(c.centroid)
            let out = c.centroid - shell.center
            let outLen = simd_length(out)
            let facing = outLen > 1e-6 ? abs(simd_dot(c.normal, out / outLen)) : 1
            let faceTerm = 1 - facingWeight + facingWeight * (facing.isFinite ? facing : 1)
            let cost = median > 0 ? min(costMax, max(costMin, c.areaPerTriangle / median)) : 1
            let w = shellTerm * sky * faceTerm * cost
            return w.isFinite && w > 0 ? w : floor * costMin
        }
    }

    static func medianAreaPerTriangle(_ cards: [SignackCard]) -> Float {
        var a = cards.map(\.areaPerTriangle).filter { $0.isFinite && $0 > 0 }
        guard !a.isEmpty else { return 0 }
        a.sort()
        return a[a.count / 2]
    }
}
