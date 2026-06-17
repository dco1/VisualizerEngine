import Foundation
import OSLog
import simd

/// Build-time **mesh soundness** auditor — the "is this mesh actually a valid
/// solid?" check a real 3D program runs for free, here as deterministic code.
///
/// Sibling to ``GeometryWinding`` (which checks only winding-vs-normals). Same
/// usage pattern: a builder that hand-authors an `SCNGeometry` calls this with
/// the positions + triangle indices it already has, BEFORE constructing the
/// geometry — no GPU read-back, so it's safe for any builder and unit-testable
/// without launching the app. In DEBUG it logs a named error per defect class;
/// in Release it's a measurement only.
///
/// It catches the mechanical mesh defects that have recurred in this project:
///   • **Out-of-range indices** — a triangle index ≥ vertex count. This is the
///     exact `docs/known-issues/illuminatorama-index-exceeds-vertexcount` bug,
///     and it renders as garbage or crashes the RT acceleration-structure build.
///   • **Non-finite vertices** — NaN / inf positions (a divide-by-zero in a
///     procedural generator), which silently corrupt bounds and culling.
///   • **Degenerate triangles** — zero-area faces (a polar-fan singularity like
///     `pepperoni-centre`, or collapsed revolution rings), which produce
///     undefined normals and shading artifacts.
///   • **Non-manifold edges** — an edge shared by more than two triangles, i.e.
///     the mesh isn't a clean surface (T-junctions, doubled faces).
///   • **Boundary edges** — edges used by exactly one triangle. Reported as
///     INFO, not a defect: a plane / ribbon / card is legitimately open. Only a
///     mesh that's *meant* to be a closed solid should have zero of these, so
///     the caller decides (`expectClosed:`).
///   • **Winding** — delegated to ``GeometryWinding`` when normals are supplied.
public enum MeshSoundness {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "MeshSoundness")

    public struct Report: Sendable {
        public var label: String
        public var vertexCount: Int
        public var triangleCount: Int
        public var outOfRangeIndices: Int
        public var nonFiniteVertices: Int
        public var degenerateTriangles: Int
        public var nonManifoldEdges: Int
        public var boundaryEdges: Int
        public var windingInconsistentFraction: Float

        /// True when no hard defect is present. Boundary edges are excluded
        /// unless `expectClosed` was set (then they fold into the verdict).
        public var isSound: Bool {
            outOfRangeIndices == 0
            && nonFiniteVertices == 0
            && degenerateTriangles == 0
            && nonManifoldEdges == 0
            && windingInconsistentFraction <= windingThreshold
            && (!expectClosed || boundaryEdges == 0)
        }

        /// One-line human summary of what's wrong (empty if sound).
        public var summary: String {
            var parts: [String] = []
            if outOfRangeIndices > 0 { parts.append("\(outOfRangeIndices) out-of-range indices") }
            if nonFiniteVertices > 0 { parts.append("\(nonFiniteVertices) NaN/inf vertices") }
            if degenerateTriangles > 0 { parts.append("\(degenerateTriangles) degenerate triangles") }
            if nonManifoldEdges > 0 { parts.append("\(nonManifoldEdges) non-manifold edges") }
            if windingInconsistentFraction > windingThreshold {
                parts.append("\(Int((windingInconsistentFraction * 100).rounded()))% inside-out winding")
            }
            if expectClosed && boundaryEdges > 0 { parts.append("\(boundaryEdges) open boundary edges (expected closed)") }
            return parts.joined(separator: ", ")
        }

        var expectClosed: Bool = false
        var windingThreshold: Float = 0.20
    }

    /// Audit a hand-built triangle mesh. `normals` is optional — when supplied,
    /// the winding-vs-normals check is run via ``GeometryWinding``. Set
    /// `expectClosed` when the mesh is meant to be a watertight solid (a vessel,
    /// a fruit), so open boundary edges count as a defect; leave it false for
    /// planes, ribbons, cards, and other legitimately-open surfaces.
    @discardableResult
    public static func audit(positions: [SIMD3<Float>],
                             indices: [UInt32],
                             normals: [SIMD3<Float>]? = nil,
                             label: String,
                             expectClosed: Bool = false,
                             windingThreshold: Float = 0.20) -> Report {
        let vertexCount = positions.count
        var report = Report(label: label, vertexCount: vertexCount, triangleCount: 0,
                            outOfRangeIndices: 0, nonFiniteVertices: 0, degenerateTriangles: 0,
                            nonManifoldEdges: 0, boundaryEdges: 0, windingInconsistentFraction: 0)
        report.expectClosed = expectClosed
        report.windingThreshold = windingThreshold

        // Non-finite vertices.
        for p in positions where !(p.x.isFinite && p.y.isFinite && p.z.isFinite) {
            report.nonFiniteVertices += 1
        }

        // Per-triangle: range, degeneracy, and edge-use tally for manifoldness.
        // Edge key = (min,max) vertex packed into a UInt64 so it's hashable and
        // direction-independent (an edge is the same regardless of which triangle
        // traverses it which way).
        var edgeUse: [UInt64: Int] = [:]
        var tri = 0
        while tri + 2 < indices.count {
            let a = indices[tri], b = indices[tri + 1], c = indices[tri + 2]
            tri += 3
            report.triangleCount += 1
            let ia = Int(a), ib = Int(b), ic = Int(c)
            guard ia < vertexCount, ib < vertexCount, ic < vertexCount else {
                report.outOfRangeIndices += 1
                continue   // can't inspect geometry of a triangle we can't index
            }
            // Degenerate: zero-area (cross product of two edges ~ 0). Also catches
            // a triangle with two identical vertices.
            let p0 = positions[ia], p1 = positions[ib], p2 = positions[ic]
            if simd_length(simd_cross(p1 - p0, p2 - p0)) < 1e-12 {
                report.degenerateTriangles += 1
            }
            tallyEdge(&edgeUse, a, b)
            tallyEdge(&edgeUse, b, c)
            tallyEdge(&edgeUse, c, a)
        }
        for (_, count) in edgeUse {
            if count == 1 { report.boundaryEdges += 1 }
            else if count > 2 { report.nonManifoldEdges += 1 }
        }

        if let normals {
            report.windingInconsistentFraction = GeometryWinding.audit(
                positions: positions, normals: normals, indices: indices,
                label: label, threshold: windingThreshold)
        }

        #if DEBUG
        if !report.isSound {
            log.error("""
            MESH SOUNDNESS '\(label, privacy: .public)': \(report.summary, privacy: .public) \
            (verts=\(vertexCount), tris=\(report.triangleCount)). A real DCC would \
            reject this mesh. See docs/known-issues/ (index-exceeds-vertexcount, \
            pepperoni-centre, scn_geometry_winding) for the recurring forms.
            """)
        }
        #endif
        return report
    }

    private static func tallyEdge(_ map: inout [UInt64: Int], _ i: UInt32, _ j: UInt32) {
        let lo = UInt64(min(i, j)), hi = UInt64(max(i, j))
        let key = (lo << 32) | hi
        map[key, default: 0] += 1
    }
}
