import Foundation
import Metal
import simd

// ── ILLUMINATORAMA CURVE PRIMITIVES (#60 item 7, increment 1) ────────────────
//
// Project-wide abstraction for ray-traced curve primitives (Metal hardware-RT
// curves, macOS 14+): branches, twigs, grass blades, stems, cables, noodles —
// anything tubular that would otherwise bloat the acceleration structure with
// swept-tube triangles. A scene registers an `IlluminatoramaCurveSet` with the
// process-wide `IlluminatoramaCurveRegistry`; the live renderer syncs the
// registry each frame, builds one curve BLAS per set
// (`MTLAccelerationStructureCurveGeometryDescriptor`), and appends one TLAS
// instance per set after the triangle instances. Curve hits then participate
// in the RT image: they occlude sun-shadow rays (both the TLAS lighting pass
// AND the surface-cache update, so cards darken under a canopy), and GI /
// reflection rays shade them with the set's material.
//
// Scope (declared, deliberate):
// - Curves exist in the RT world only. Primary visibility is the rastered
//   G-buffer, so a scene keeps (cheap) triangle tubes for the visible image —
//   the win is that the ACCELERATION STRUCTURE no longer pays triangle cost,
//   which is exactly Forest's 150k-triangle-cap problem (#60 item 7).
// - Catmull-Rom basis, round cross-section in increment 1 (round curves have
//   exact 3D normals — right for branches viewed close; the flat/hair variant
//   is a follow-on). The registry validates and rejects anything else.
// - Static placement per set (`transform` is rigid; captured at registration).
//   Live/deforming control points (solver-written buffer + per-frame refit,
//   the same trick as the deforming triangle BLAS) are the documented next
//   increment.
// - Curve surfaces are not surface-cache cards: a GI ray that hits a curve
//   re-shades it directly (sun + ambient), and the cache-update kernel treats
//   a curve hit as an occluder.

/// One renderable set of round Catmull-Rom curve primitives.
///
/// `controlPoints[i]` pairs with `radii[i]`. Each entry of `segmentIndices`
/// is the index of the FIRST of 4 consecutive control points forming one
/// curve segment (Metal's Catmull-Rom convention: the curve passes through
/// P1..P2; P0/P3 steer the end tangents). Consecutive segments of one strand
/// overlap 3 control points (`segmentIndices` advancing by 1).
public final class IlluminatoramaCurveSet: Sendable {
    public struct Material: Sendable {
        public var albedo: SIMD3<Float>
        public var roughness: Float
        public var emission: SIMD3<Float>
        public init(albedo: SIMD3<Float>, roughness: Float = 0.7,
                    emission: SIMD3<Float> = .zero) {
            self.albedo = albedo; self.roughness = roughness; self.emission = emission
        }
    }

    public let controlPoints: [SIMD3<Float>]
    public let radii: [Float]
    public let segmentIndices: [UInt32]
    public let material: Material
    /// Rigid world placement of the whole set (object→world). Captured at
    /// registration. For a deforming set the control points themselves are
    /// re-displaced per frame (see `windAttr`); a rigid set keeps this transform.
    public let transform: simd_float4x4
    /// Per-control-point wind attribute `(swayWeight, phase, flutter, _)`
    /// (#60 item 7, increment 2). Empty ⇒ a RIGID set (no per-frame sway). When
    /// present (count == controlPoints.count), the renderer re-displaces the
    /// control points each frame with the SAME `applyTreeWind` the G-buffer
    /// vertex shader applies to the rastered wood, then refits the curve geometry
    /// — so the RT curve shadows track the swaying tubes. Encoding matches the
    /// vertex tangent: x=swayWeight (height²), y=tree phase, z=flutter.
    public let windAttr: [SIMD4<Float>]
    public let label: String

    /// Returns nil when the data is inconsistent (mismatched counts, a segment
    /// reaching past the control points, or a negative radius) — the registry
    /// never accepts an un-traceable set.
    public init?(controlPoints: [SIMD3<Float>],
                 radii: [Float],
                 segmentIndices: [UInt32],
                 material: Material,
                 transform: simd_float4x4 = matrix_identity_float4x4,
                 windAttr: [SIMD4<Float>] = [],
                 label: String = "curveSet") {
        guard controlPoints.count >= 4,
              radii.count == controlPoints.count,
              !segmentIndices.isEmpty,
              radii.allSatisfy({ $0 >= 0 }),
              segmentIndices.allSatisfy({ Int($0) + 4 <= controlPoints.count }),
              windAttr.isEmpty || windAttr.count == controlPoints.count
        else { return nil }
        self.controlPoints = controlPoints
        self.radii = radii
        self.segmentIndices = segmentIndices
        self.material = material
        self.transform = transform
        self.windAttr = windAttr
        self.label = label
    }
}

/// Process-wide registry connecting scenes (which don't hold a renderer
/// reference — the overlay owns it) to the live renderer. Same singleton
/// pattern as `IlluminatoramaSharedSettings`. The renderer compares `version`
/// once per frame and re-syncs (→ AS rebuild) when it changed; scenes
/// register on build and unregister on teardown.
@MainActor
public final class IlluminatoramaCurveRegistry {
    public static let shared = IlluminatoramaCurveRegistry()
    private init() {}

    public private(set) var sets: [IlluminatoramaCurveSet] = []
    /// Bumped on every mutation; the renderer's per-frame sync key.
    public private(set) var version: Int = 1

    public func register(_ set: IlluminatoramaCurveSet) {
        guard !sets.contains(where: { $0 === set }) else { return }
        sets.append(set)
        version += 1
    }

    public func unregister(_ set: IlluminatoramaCurveSet) {
        let before = sets.count
        sets.removeAll { $0 === set }
        if sets.count != before { version += 1 }
    }

    public func removeAll() {
        guard !sets.isEmpty else { return }
        sets.removeAll()
        version += 1
    }
}
