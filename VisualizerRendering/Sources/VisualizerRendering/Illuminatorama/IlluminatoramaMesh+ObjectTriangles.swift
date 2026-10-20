import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaMesh {

    // ── CPU object-space triangle soup (for RT acceleration structure) ─────────

    /// Object-space triangle soup for one element: positions + a flat `UInt32`
    /// index list. The RT path uses this to bake a world-space triangle soup
    /// from extracted scene geometry (the caller applies the node's world
    /// transform). Returns nil for GPU-backed / non-triangle / empty elements —
    /// the same elements `from(...)` already refuses. `SCNFloor` is emitted as a
    /// large XZ quad, matching `groundQuad`'s deferred-pass stand-in.
    public static func objectTriangles(scnGeometry geometry: SCNGeometry,
                                        elementIndex: Int)
        -> (positions: [SIMD3<Float>], indices: [UInt32])? {
        if geometry is SCNFloor {
            let s: Float = 400
            return ([SIMD3(-s, 0, -s), SIMD3(s, 0, -s), SIMD3(s, 0, s), SIMD3(-s, 0, s)],
                    [0, 1, 2, 0, 2, 3])
        }
        guard elementIndex < geometry.elements.count else { return nil }
        let element = geometry.elements[elementIndex]
        guard element.primitiveType == .triangles,
              let rawPositions = floatTriples(from: geometry.sources(for: .vertex).first),
              !rawPositions.isEmpty, element.primitiveCount > 0,
              let idx = readIndexData(element: element) else { return nil }
        // Same unit-primitive correction as `from(scnGeometry:)` — the RT soup
        // must agree with the deferred G-buffer mesh or a box's raster draw and
        // its ray-traced twin would be different sizes.
        let positions = normaliseToBoundingBox(rawPositions, geometry: geometry)
        var indices = [UInt32](); indices.reserveCapacity(idx.count)
        idx.data.withUnsafeBytes { raw in
            switch idx.type {
            case .uint16:
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0..<idx.count { indices.append(UInt32(p[i])) }
            case .uint32:
                let p = raw.bindMemory(to: UInt32.self)
                for i in 0..<idx.count { indices.append(p[i]) }
            @unknown default: break
            }
        }
        guard !indices.isEmpty else { return nil }
        return (positions, indices)
    }

    /// Per-triangle OBJECT-space geometric normals (`float4`, w = 0), computed
    /// from this mesh's own (shared-storage) vertex + index buffers. Built once
    /// when the RT BLAS is created; the instanced RT kernel transforms each by
    /// the per-instance normal matrix to get the world normal at a hit. Returns
    /// `[]` if the buffers aren't CPU-readable.
    public func objectFaceNormals() -> [SIMD4<Float>] {
        let triCount = indexCount / 3
        guard triCount > 0 else { return [] }
        // GPU-direct meshes (.storageModePrivate) have no CPU-readable backing;
        // contents() would abort under the Metal debug layer.
        guard vertexBuffer.storageMode != .private else { return [] }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        let vcount = vertexCount
        let vbase = vertexBuffer.contents()
        let ibase = indexBuffer.contents()
        func pos(_ i: Int) -> SIMD3<Float> {
            vbase.load(fromByteOffset: i * stride, as: SIMD3<Float>.self)
        }
        var out = [SIMD4<Float>](); out.reserveCapacity(triCount)
        for t in 0..<triCount {
            let i0: Int, i1: Int, i2: Int
            if indexType == .uint16 {
                let o = t * 3 * 2
                i0 = Int(ibase.load(fromByteOffset: o,     as: UInt16.self))
                i1 = Int(ibase.load(fromByteOffset: o + 2, as: UInt16.self))
                i2 = Int(ibase.load(fromByteOffset: o + 4, as: UInt16.self))
            } else {
                let o = t * 3 * 4
                i0 = Int(ibase.load(fromByteOffset: o,     as: UInt32.self))
                i1 = Int(ibase.load(fromByteOffset: o + 4, as: UInt32.self))
                i2 = Int(ibase.load(fromByteOffset: o + 8, as: UInt32.self))
            }
            // Guard against an index that reaches past the readable vertex block.
            // from()-registered meshes are sanitised (sanitiseIndices), but
            // descriptor / setRTGeometry meshes set indexBuffer + vertexCount
            // independently and bypass that path; an out-of-range index here is a
            // raw out-of-bounds pointer load — worse than an array trap (UB /
            // garbage normals), and the load() can't be caught. Emit a placeholder
            // normal so `out` stays length == triCount and primitive_id alignment
            // with the BLAS is preserved. See
            // docs/known-issues/illuminatorama-index-exceeds-vertexcount.md.
            guard i0 < vcount, i1 < vcount, i2 < vcount else {
                out.append(SIMD4<Float>(0, 1, 0, 0)); continue
            }
            let a = pos(i0), b = pos(i1), c = pos(i2)
            var n = simd_cross(b - a, c - a)
            let len = simd_length(n)
            n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
            out.append(SIMD4<Float>(n, 0))
        }
        return out
    }

    /// Per-triangle MESH UVs — the same `(uvA, uvB, uvC)` the rasteriser
    /// interpolates in `illumi_fs` — flattened to 3 entries per triangle, in the
    /// EXACT triangle order `objectFaceNormals()` uses. A ray-traced hit therefore
    /// reads its texture coordinates at `uv[(normalBase + primitive_id) * 3 + k]`,
    /// the same `normalBase + prim` indexing the normals already use, and can
    /// barycentric-interpolate a real atlas sample instead of collapsing the
    /// surface to the instance's mean albedo (docs/known-issues.md
    /// "Through-glass scenery is UNTEXTURED"). Returns `[]` if the buffers aren't
    /// CPU-readable — the shader falls back to the mean albedo in that case.
    public func objectFaceUVs() -> [SIMD2<Float>] {
        let triCount = indexCount / 3
        guard triCount > 0 else { return [] }
        guard vertexBuffer.storageMode != .private else { return [] }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        // uv sits after position (float3 + pad) and normal (float3 + pad).
        let uvOffset = MemoryLayout<IlluminatoramaVertex>.offset(of: \.uv) ?? 32
        let vcount = vertexCount
        let vbase = vertexBuffer.contents()
        let ibase = indexBuffer.contents()
        var out = [SIMD2<Float>](); out.reserveCapacity(triCount * 3)
        for t in 0..<triCount {
            var idx = (0, 0, 0)
            if indexType == .uint16 {
                let o = t * 3 * 2
                idx = (Int(ibase.load(fromByteOffset: o,     as: UInt16.self)),
                       Int(ibase.load(fromByteOffset: o + 2, as: UInt16.self)),
                       Int(ibase.load(fromByteOffset: o + 4, as: UInt16.self)))
            } else {
                let o = t * 3 * 4
                idx = (Int(ibase.load(fromByteOffset: o,     as: UInt32.self)),
                       Int(ibase.load(fromByteOffset: o + 4, as: UInt32.self)),
                       Int(ibase.load(fromByteOffset: o + 8, as: UInt32.self)))
            }
            // Same out-of-range guard as objectFaceNormals: a raw pointer load past
            // the readable vertex block is UB, and the array MUST stay exactly
            // 3×triCount long or every later triangle's UVs shift.
            guard idx.0 < vcount, idx.1 < vcount, idx.2 < vcount else {
                out.append(.zero); out.append(.zero); out.append(.zero); continue
            }
            for i in [idx.0, idx.1, idx.2] {
                out.append(vbase.load(fromByteOffset: i * stride + uvOffset, as: SIMD2<Float>.self))
            }
        }
        return out
    }

    /// Per-triangle-CORNER object-space vertex normals — 3 per triangle, in the EXACT
    /// triangle order `objectFaceNormals()` uses — so a ray-traced hit can barycentric-
    /// interpolate the SHADING normal the rasteriser uses instead of the flat face normal.
    ///
    /// The glass bounce loop needs this and nothing else does: a dielectric boundary's
    /// normal IS its refraction, so a smooth-shaded moulded pane (a glass block's pillow,
    /// a ripple) traced against its face normals refracts as a set of flat prisms — every
    /// facet prints its own sharp, displaced copy of the scene behind it. Returns `[]` if the
    /// buffers aren't CPU-readable; the shader then falls back to the face normal.
    public func objectCornerNormals() -> [SIMD4<Float>] {
        let triCount = indexCount / 3
        guard triCount > 0 else { return [] }
        guard vertexBuffer.storageMode != .private else { return [] }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        let nOffset = MemoryLayout<IlluminatoramaVertex>.offset(of: \.normal) ?? 16
        let vcount = vertexCount
        let vbase = vertexBuffer.contents()
        let ibase = indexBuffer.contents()
        var out = [SIMD4<Float>](); out.reserveCapacity(triCount * 3)
        for t in 0..<triCount {
            var idx = (0, 0, 0)
            if indexType == .uint16 {
                let o = t * 3 * 2
                idx = (Int(ibase.load(fromByteOffset: o,     as: UInt16.self)),
                       Int(ibase.load(fromByteOffset: o + 2, as: UInt16.self)),
                       Int(ibase.load(fromByteOffset: o + 4, as: UInt16.self)))
            } else {
                let o = t * 3 * 4
                idx = (Int(ibase.load(fromByteOffset: o,     as: UInt32.self)),
                       Int(ibase.load(fromByteOffset: o + 4, as: UInt32.self)),
                       Int(ibase.load(fromByteOffset: o + 8, as: UInt32.self)))
            }
            // Same guard as objectFaceUVs: the array MUST stay exactly 3×triCount long. A zero
            // normal is the shader's "use the face normal" signal.
            guard idx.0 < vcount, idx.1 < vcount, idx.2 < vcount else {
                out.append(.zero); out.append(.zero); out.append(.zero); continue
            }
            for i in [idx.0, idx.1, idx.2] {
                let n = vbase.load(fromByteOffset: i * stride + nOffset, as: SIMD3<Float>.self)
                out.append(SIMD4<Float>(n, 0))
            }
        }
        return out
    }

    /// Object-space positions + flat `UInt32` index list read straight from this
    /// mesh's CPU-readable shared-storage buffers (same access pattern as
    /// `objectFaceNormals`). The surface-cache TLAS path (P1c) uses this to bake
    /// a per-instance world-space soup in the SAME grouped order the TLAS
    /// enumerates instances, so a hit's `(instance_id, primitive_id)` maps to a
    /// global soup triangle via `soupTriBase[instance_id] + primitive_id`.
    public func objectTriangleSoup() -> (positions: [SIMD3<Float>], indices: [UInt32]) {
        guard vertexBuffer.storageMode != .private else { return ([], []) }
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        let vbase = vertexBuffer.contents()
        var positions = [SIMD3<Float>](); positions.reserveCapacity(vertexCount)
        for v in 0..<vertexCount {
            positions.append(vbase.load(fromByteOffset: v * stride, as: SIMD3<Float>.self))
        }
        let ibase = indexBuffer.contents()
        var indices = [UInt32](); indices.reserveCapacity(indexCount)
        if indexType == .uint16 {
            for i in 0..<indexCount {
                indices.append(UInt32(ibase.load(fromByteOffset: i * 2, as: UInt16.self)))
            }
        } else {
            for i in 0..<indexCount {
                indices.append(ibase.load(fromByteOffset: i * 4, as: UInt32.self))
            }
        }
        return (positions, indices)
    }
}
