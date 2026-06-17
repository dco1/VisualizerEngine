import Foundation
import OSLog
import simd

/// Runtime guard against the recurring **custom-`SCNGeometry` winding bug**:
/// triangle winding that disagrees with the authored per-vertex normals, so
/// single-sided back-face culling draws *nothing* where a surface should be
/// (an "inside-out" mesh). This is the class of bug that made the ChipsAndGuac
/// molcajete's interior invisible — you saw straight through the bowl, and the
/// guac inside had no visible containing wall (read as "bleeding").
///
/// The geometric test is exact and cheap: for each triangle, the winding normal
/// `Ng = (p1−p0) × (p2−p0)` must point the *same way* as the averaged vertex
/// normal `Na = n0+n1+n2`. If `dot(Ng, Na) < 0`, the winding and the normal
/// disagree — SceneKit culls by winding, so that face vanishes (or shades
/// inverted) under a single-sided material.
///
/// Call ``audit(positions:normals:indices:label:)`` from any builder that
/// hand-authors an `SCNGeometry` (positions + normals + a triangle index list)
/// BEFORE constructing the geometry — it has the arrays right there. In DEBUG it
/// logs a loud, named error when a mesh is inside-out; in Release it's a no-op
/// (returns the fraction without logging). See
/// `docs/known-issues/scn_geometry_winding.md`.
///
/// Enforced by `Scripts/winding-stop-hook.sh`: any new hand-built `SCNGeometry`
/// in a scene/asset file must either call this auditor or carry a
/// `// winding-ok: <reason>` suppression.
public enum GeometryWinding {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "GeometryWinding")

    /// Fraction (0…1) of triangles whose winding disagrees with their vertex
    /// normals. 0 = consistent (renders correctly single-sided). A value near
    /// 1 = the whole mesh is inside-out. In DEBUG, logs an error naming `label`
    /// when the fraction exceeds `threshold`.
    ///
    /// Degenerate triangles (zero-area) and vertices with a zero normal are
    /// skipped — they can't be classified and shouldn't inflate the ratio.
    @discardableResult
    public static func audit(positions: [SIMD3<Float>],
                             normals: [SIMD3<Float>],
                             indices: [UInt32],
                             label: String,
                             threshold: Float = 0.20) -> Float {
        guard indices.count >= 3, positions.count == normals.count, !positions.isEmpty else {
            return 0
        }
        let vertexCount = positions.count
        var disagree = 0
        var classified = 0
        var tri = 0
        while tri + 2 < indices.count {
            let ia = Int(indices[tri]); let ib = Int(indices[tri + 1]); let ic = Int(indices[tri + 2])
            tri += 3
            guard ia < vertexCount, ib < vertexCount, ic < vertexCount else { continue }
            let p0 = positions[ia], p1 = positions[ib], p2 = positions[ic]
            let ng = simd_cross(p1 - p0, p2 - p0)
            let na = normals[ia] + normals[ib] + normals[ic]
            let lng = simd_length(ng), lna = simd_length(na)
            if lng < 1e-12 || lna < 1e-12 { continue }   // degenerate / no authored normal
            classified += 1
            if simd_dot(ng, na) < 0 { disagree += 1 }
        }
        guard classified > 0 else { return 0 }
        let frac = Float(disagree) / Float(classified)

        #if DEBUG
        if frac > threshold {
            log.error("""
            WINDING ⊥ NORMALS in '\(label, privacy: .public)': \
            \(disagree)/\(classified) (\(Int((frac * 100).rounded()))%) triangles are \
            inside-out — their winding disagrees with their vertex normals, so a \
            single-sided material will CULL these faces (invisible interior / \
            "bleeding" contents). Fix: swap the last two indices of the affected \
            triangles so winding matches the normal, or set the material \
            double-sided for a true two-sided surface. \
            See docs/known-issues/scn_geometry_winding.md
            """)
        } else if frac > 0 {
            log.debug("winding audit '\(label, privacy: .public)': \(disagree)/\(classified) tris inconsistent (under threshold)")
        }
        #endif
        return frac
    }
}
