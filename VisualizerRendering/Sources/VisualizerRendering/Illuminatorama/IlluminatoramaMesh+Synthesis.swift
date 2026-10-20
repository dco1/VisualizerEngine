import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaMesh {

    // ── Synthesised normals ───────────────────────────────────────────────────

    /// When the source mesh ships positions but no normals (common for
    /// hand-built `SCNGeometrySource` content), compute smooth vertex
    /// normals by averaging the face normals of every triangle each vertex
    /// participates in. Cheap one-pass over the index buffer.
    static func synthesiseNormals(into verts: [IlluminatoramaVertex],
                                           indexData: Data,
                                           indexCount: Int,
                                           indexType: MTLIndexType) -> [IlluminatoramaVertex] {
        var accum = Array(repeating: SIMD3<Float>.zero, count: verts.count)
        indexData.withUnsafeBytes { raw in
            switch indexType {
            case .uint16:
                let idx = raw.bindMemory(to: UInt16.self)
                accumulate(triangleCount: indexCount / 3, getIndex: { i in Int(idx[i]) },
                           verts: verts, into: &accum)
            case .uint32:
                let idx = raw.bindMemory(to: UInt32.self)
                accumulate(triangleCount: indexCount / 3, getIndex: { i in Int(idx[i]) },
                           verts: verts, into: &accum)
            @unknown default:
                break
            }
        }
        var out = verts
        for i in 0..<out.count {
            let n = accum[i]
            let len = simd_length(n)
            out[i] = IlluminatoramaVertex(
                position: verts[i].position,
                normal: len > 1e-6 ? n / len : SIMD3<Float>(0, 1, 0),
                uv: verts[i].uv
            )
        }
        return out
    }

    // ── Synthesised tangents ──────────────────────────────────────────────────

    /// Standard derivative-of-UV tangent synthesis. For each triangle with
    /// positions (p0, p1, p2) and UVs (uv0, uv1, uv2), the partial
    /// derivative of position w.r.t. u (in tangent space) is:
    ///
    ///     T = (duv2.y * edge1 - duv1.y * edge2) / det
    ///
    /// Accumulate T per vertex and orthonormalise against the vertex
    /// normal — produces a stable TBN frame for shading normal maps.
    /// Handedness is left at +1; mirrored-UV islands would need a per-
    /// triangle sign computation, which would matter for those islands
    /// but is out of scope here. Most non-mirrored meshes look correct
    /// with the +1 default.
    static func synthesiseTangents(into verts: [IlluminatoramaVertex],
                                            indexData: Data,
                                            indexCount: Int,
                                            indexType: MTLIndexType) -> [IlluminatoramaVertex] {
        var accum = Array(repeating: SIMD3<Float>.zero, count: verts.count)
        indexData.withUnsafeBytes { raw in
            switch indexType {
            case .uint16:
                let idx = raw.bindMemory(to: UInt16.self)
                accumulateTangents(triangleCount: indexCount / 3,
                                    getIndex: { i in Int(idx[i]) },
                                    verts: verts, into: &accum)
            case .uint32:
                let idx = raw.bindMemory(to: UInt32.self)
                accumulateTangents(triangleCount: indexCount / 3,
                                    getIndex: { i in Int(idx[i]) },
                                    verts: verts, into: &accum)
            @unknown default:
                break
            }
        }
        var out = verts
        for i in 0..<out.count {
            let n = out[i].normal
            let raw = accum[i]
            // Gram-Schmidt: project T onto the plane orthogonal to N so
            // the TBN frame is orthogonal. Falls back to a stable
            // arbitrary tangent when the accumulator is degenerate (no
            // triangle contributed a derivative — typical for unwelded
            // single-vertex islands or zero-UV-area triangles).
            var t = raw - n * simd_dot(raw, n)
            let len = simd_length(t)
            if len < 1e-5 {
                // Arbitrary tangent perpendicular to N — pick whichever
                // axis is least aligned.
                t = abs(n.y) < 0.99 ? simd_cross(SIMD3(0, 1, 0), n)
                                     : simd_cross(SIMD3(1, 0, 0), n)
                t = simd_normalize(t)
            } else {
                t /= len
            }
            out[i].tangent = SIMD4(t, 1.0)
        }
        return out
    }

    static func accumulateTangents(triangleCount: Int,
                                            getIndex: (Int) -> Int,
                                            verts: [IlluminatoramaVertex],
                                            into accum: inout [SIMD3<Float>]) {
        for t in 0..<triangleCount {
            let i0 = getIndex(t * 3)
            let i1 = getIndex(t * 3 + 1)
            let i2 = getIndex(t * 3 + 2)
            guard i0 < verts.count, i1 < verts.count, i2 < verts.count else { continue }
            let p0 = verts[i0].position
            let p1 = verts[i1].position
            let p2 = verts[i2].position
            let uv0 = verts[i0].uv
            let uv1 = verts[i1].uv
            let uv2 = verts[i2].uv
            let edge1 = p1 - p0
            let edge2 = p2 - p0
            let duv1 = uv1 - uv0
            let duv2 = uv2 - uv0
            let det = duv1.x * duv2.y - duv1.y * duv2.x
            if abs(det) < 1e-8 { continue }
            let inv = 1.0 / det
            let T = (edge1 * duv2.y - edge2 * duv1.y) * inv
            accum[i0] += T
            accum[i1] += T
            accum[i2] += T
        }
    }

    private static func accumulate(triangleCount: Int,
                                    getIndex: (Int) -> Int,
                                    verts: [IlluminatoramaVertex],
                                    into accum: inout [SIMD3<Float>]) {
        for t in 0..<triangleCount {
            let i0 = getIndex(t * 3)
            let i1 = getIndex(t * 3 + 1)
            let i2 = getIndex(t * 3 + 2)
            guard i0 < verts.count, i1 < verts.count, i2 < verts.count else { continue }
            let p0 = verts[i0].position
            let p1 = verts[i1].position
            let p2 = verts[i2].position
            let n = simd_cross(p1 - p0, p2 - p0)
            // Don't normalise here: a triangle's contribution scales with
            // its area, which is the standard area-weighted-average
            // normals scheme. Normalisation happens in the caller.
            accum[i0] += n
            accum[i1] += n
            accum[i2] += n
        }
    }
}
