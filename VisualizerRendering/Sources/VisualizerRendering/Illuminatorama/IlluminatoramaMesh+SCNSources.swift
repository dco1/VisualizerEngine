import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaMesh {

    // ── SCNGeometrySource → [SIMD3<Float>] / [SIMD2<Float>] ──────────────────

    /// Reads a vertex source with 3 floats per element. Tolerates double
    /// components (rare but legal — SceneKit can emit Doubles on Intel)
    /// and any positive `dataOffset` / `dataStride`. Returns nil for
    /// non-float components or unsupported widths.
    static func floatTriples(from source: SCNGeometrySource?) -> [SIMD3<Float>]? {
        guard let s = source else { return nil }
        guard s.componentsPerVector >= 3, s.usesFloatComponents else { return nil }
        let bpc = s.bytesPerComponent
        let stride = s.dataStride
        let offset = s.dataOffset
        let count = s.vectorCount
        guard count > 0, bpc == 4 || bpc == 8 else { return nil }
        // MTLBuffer-backed sources surface a Data that doesn't reflect the
        // GPU contents — usually empty or too short for vectorCount*stride.
        // Bail rather than walk an unsafe pointer off the end.
        let neededBytes = offset + (count - 1) * stride + 3 * bpc
        guard s.data.count >= neededBytes else { return nil }
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(count)
        s.data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<count {
                let p = base.advanced(by: offset + i * stride)
                if bpc == 4 {
                    let f = p.assumingMemoryBound(to: Float.self)
                    out.append(SIMD3<Float>(f[0], f[1], f[2]))
                } else {
                    let d = p.assumingMemoryBound(to: Double.self)
                    out.append(SIMD3<Float>(Float(d[0]), Float(d[1]), Float(d[2])))
                }
            }
        }
        return out.count == count ? out : nil
    }

    /// Phase 4.17 — pull a per-vertex `SCNGeometrySource(semantic: .color)`
    /// into a flat `[SIMD4<Float>]`. Handles the two layouts SceneKit
    /// actually emits: float RGBA (`bpc==4`, `componentsPerVector==4`)
    /// — the path `SCNGeometrySource(data:semantic:.color, …,
    /// usesFloatComponents:true, componentsPerVector:4, bytesPerComponent:4)`
    /// takes — and the legacy `usesFloatComponents:false` byte path some
    /// imported assets ship. Float RGB (`componentsPerVector==3`) is also
    /// accepted, alpha defaulting to 1. Returns nil for unsupported widths
    /// rather than guessing — callers fall through to the default-white
    /// path so a mesh without colours stays a no-op multiply at shading
    /// time.
    static func floatColors(from source: SCNGeometrySource?) -> [SIMD4<Float>]? {
        guard let s = source else { return nil }
        let comps = s.componentsPerVector
        let bpc = s.bytesPerComponent
        let stride = s.dataStride
        let offset = s.dataOffset
        let count = s.vectorCount
        guard count > 0, comps >= 3, comps <= 4 else { return nil }
        let bytesPerVector = comps * bpc
        let neededBytes = offset + (count - 1) * stride + bytesPerVector
        guard s.data.count >= neededBytes else { return nil }
        var out: [SIMD4<Float>] = []
        out.reserveCapacity(count)
        s.data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<count {
                let p = base.advanced(by: offset + i * stride)
                if s.usesFloatComponents {
                    if bpc == 4 {
                        let f = p.assumingMemoryBound(to: Float.self)
                        let a = comps >= 4 ? f[3] : 1
                        out.append(SIMD4<Float>(f[0], f[1], f[2], a))
                    } else if bpc == 8 {
                        let d = p.assumingMemoryBound(to: Double.self)
                        let a = comps >= 4 ? Float(d[3]) : 1
                        out.append(SIMD4<Float>(Float(d[0]), Float(d[1]),
                                                 Float(d[2]), a))
                    }
                } else if bpc == 1 {
                    // Unsigned-byte RGBA (the path SCNGeometrySource(colors:)
                    // takes when fed an [SCNVector4]-of-bytes asset import).
                    // 0..255 → 0..1 linearly.
                    let b = p.assumingMemoryBound(to: UInt8.self)
                    let a = comps >= 4 ? Float(b[3]) / 255.0 : 1
                    out.append(SIMD4<Float>(Float(b[0]) / 255.0,
                                             Float(b[1]) / 255.0,
                                             Float(b[2]) / 255.0,
                                             a))
                }
            }
        }
        return out.count == count ? out : nil
    }

    static func floatPairs(from source: SCNGeometrySource?) -> [SIMD2<Float>]? {
        guard let s = source else { return nil }
        guard s.componentsPerVector >= 2, s.usesFloatComponents else { return nil }
        let bpc = s.bytesPerComponent
        let stride = s.dataStride
        let offset = s.dataOffset
        let count = s.vectorCount
        guard count > 0, bpc == 4 || bpc == 8 else { return nil }
        // Same bound check as floatTriples — GPU-backed sources can ship a
        // CPU `data` that's shorter than `vectorCount * stride`.
        let neededBytes = offset + (count - 1) * stride + 2 * bpc
        guard s.data.count >= neededBytes else { return nil }
        var out: [SIMD2<Float>] = []
        out.reserveCapacity(count)
        s.data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for i in 0..<count {
                let p = base.advanced(by: offset + i * stride)
                if bpc == 4 {
                    let f = p.assumingMemoryBound(to: Float.self)
                    out.append(SIMD2<Float>(f[0], f[1]))
                } else {
                    let d = p.assumingMemoryBound(to: Double.self)
                    out.append(SIMD2<Float>(Float(d[0]), Float(d[1])))
                }
            }
        }
        return out.count == count ? out : nil
    }

    /// A box-projected UV for one vertex with no authored texcoord: pick the plane orthogonal to
    /// the DOMINANT axis of `normal` and use the other two position components, directly in the
    /// mesh's own local-space metres. That matches — deliberately, with no extra scale factor —
    /// the "one texture repeat per metre" convention every instanced placeable's own procedurally-
    /// generated mesh already bakes its UV at (`ElementUVScale.furnitureSlice`, `SurfaceTiling
    /// .prebaked(1.0)`): a custom object with no UV of its own now tiles a picked material at the
    /// same real-world scale a piece of built-in furniture does, rather than at an arbitrary one.
    ///
    /// This is a hard per-vertex projection, not blended triplanar sampling: two faces meeting
    /// across a dominant-axis boundary can show a seam in the texture, same as it would with any
    /// single-sample box projection. That is the honest cost of a real per-pixel texture on a
    /// surface the source file never gave a UV unwrap — strictly better than the flat, textureless
    /// colour every vertex sharing `(0, 0)` produced before.
    static func triplanarUV(position p: SIMD3<Float>, normal n: SIMD3<Float>) -> SIMD2<Float> {
        let a = simd_abs(n)
        if a.x >= a.y, a.x >= a.z { return SIMD2(p.z, p.y) }   // dominant ±X: project onto ZY
        if a.y >= a.x, a.y >= a.z { return SIMD2(p.x, p.z) }   // dominant ±Y: project onto XZ
        return SIMD2(p.x, p.y)                                  // dominant ±Z: project onto XY
    }

    // ── Index conversion ──────────────────────────────────────────────────────

    /// DH-0742 — expand a "split-vertex" element (see the call site's comment) into a
    /// flat per-corner vertex list plus a trivial sequential index buffer. `streamsPerCorner`
    /// interleaved index values per triangle corner are de-interleaved into that many
    /// candidate streams; each attribute source (position/normal/uv/color) is matched to
    /// whichever stream its OWN values validly index into (every value `< source.count`),
    /// preferring the tightest fit so a smaller source's range can't be mismatched onto a
    /// wider stream that happens to also cover it. Returns `nil` only if no stream fits the
    /// position source — the one attribute every mesh must have.
    static func expandSplitVertexElement(
        element: SCNGeometryElement, cornerCount: Int, streamsPerCorner: Int,
        positions: [SIMD3<Float>], normals: [SIMD3<Float>]?, uvs: [SIMD2<Float>]?,
        colors: [SIMD4<Float>]?, geometryLabel: String
    ) -> (verts: [IlluminatoramaVertex], needsSynthNormals: Bool,
          indexData: Data, indexCount: Int)? {
        let bpi = element.bytesPerIndex
        let totalEntries = cornerCount * streamsPerCorner
        var raw: [UInt32] = []
        raw.reserveCapacity(totalEntries)
        element.data.withUnsafeBytes { buf in
            switch bpi {
            case 1:
                let p = buf.bindMemory(to: UInt8.self)
                guard p.count >= totalEntries else { return }
                for i in 0..<totalEntries { raw.append(UInt32(p[i])) }
            case 2:
                let p = buf.bindMemory(to: UInt16.self)
                guard p.count >= totalEntries else { return }
                for i in 0..<totalEntries { raw.append(UInt32(p[i])) }
            case 4:
                let p = buf.bindMemory(to: UInt32.self)
                guard p.count >= totalEntries else { return }
                for i in 0..<totalEntries { raw.append(p[i]) }
            default:
                break
            }
        }
        guard raw.count == totalEntries else {
            log.debug("Illuminatorama mesh \(geometryLabel): split-vertex element — couldn't read \(totalEntries) index entries at \(bpi) bytes each; skipping")
            return nil
        }

        // De-interleave into `streamsPerCorner` per-corner arrays.
        var streams = Array(repeating: [UInt32](repeating: 0, count: cornerCount), count: streamsPerCorner)
        for k in 0..<cornerCount {
            for s in 0..<streamsPerCorner { streams[s][k] = raw[k * streamsPerCorner + s] }
        }

        // Tightest fit = the stream whose max value is CLOSEST to (but still under) the
        // source's own count — i.e. the LARGEST valid max, not the smallest. A smaller
        // source's index range (e.g. position, welded to fewer unique points) can be
        // coincidentally in-range for a larger source's slot too (every value < count still
        // passes the bounds check), so picking the smallest max would wrongly prefer the
        // narrower stream and under-utilize the wider source it doesn't actually belong to.
        func bestStream(for count: Int) -> Int? {
            var best: Int? = nil
            var bestMax: Int = -1
            for (s, values) in streams.enumerated() {
                guard let m = values.max(), Int(m) < count else { continue }
                if Int(m) > bestMax { bestMax = Int(m); best = s }
            }
            return best
        }
        guard let posStream = bestStream(for: positions.count) else {
            log.debug("Illuminatorama mesh \(geometryLabel): split-vertex element — no index stream fits the position source (count \(positions.count)); skipping")
            return nil
        }
        let normalStream = normals.flatMap { bestStream(for: $0.count) }
        let uvStream = uvs.flatMap { bestStream(for: $0.count) }
        let colorStream = colors.flatMap { bestStream(for: $0.count) }

        var verts: [IlluminatoramaVertex] = []
        verts.reserveCapacity(cornerCount)
        for k in 0..<cornerCount {
            let p = positions[Int(streams[posStream][k])]
            let n = normalStream.map { normals![Int(streams[$0][k])] } ?? SIMD3<Float>(0, 1, 0)
            // No texcoord stream at all: same box-projected fallback as the shared-index path
            // above, not a shared (0,0) — see the header comment on `from(scnGeometry:...)`.
            let uv = uvStream.map { uvs![Int(streams[$0][k])] } ?? triplanarUV(position: p, normal: n)
            let c = colorStream.map { colors![Int(streams[$0][k])] } ?? SIMD4<Float>(1, 1, 1, 1)
            verts.append(IlluminatoramaVertex(position: p, normal: n, uv: uv, color: c))
        }
        let indexArray: [UInt32] = (0..<cornerCount).map { UInt32($0) }
        let indexData = indexArray.withUnsafeBufferPointer { Data(buffer: $0) }
        return (verts, normals == nil, indexData, cornerCount)
    }

    /// Pull the raw index `Data` blob out of an `SCNGeometryElement`, promote
    /// 1-byte indices to 2-byte (Metal has no `MTLIndexType.uint8`), and
    /// return a tuple ready to feed `init(device:vertices:indices:indexCount:indexType:)`.
    static func readIndexData(element: SCNGeometryElement)
        -> (data: Data, count: Int, type: MTLIndexType)? {
        let count = element.primitiveCount * 3
        let bpi = element.bytesPerIndex
        let raw = element.data
        // GPU-backed elements report a nonzero primitiveCount but ship empty
        // Data — there's no CPU buffer to copy. Refuse them here so callers
        // see a clean nil rather than a zero-length Metal buffer.
        guard count > 0, !raw.isEmpty else { return nil }
        switch bpi {
        case 1:
            // Promote: walk each byte and write a UInt16 for it.
            var promoted = Data(count: count * MemoryLayout<UInt16>.size)
            promoted.withUnsafeMutableBytes { outRaw in
                let out = outRaw.bindMemory(to: UInt16.self)
                raw.withUnsafeBytes { inRaw in
                    let bytes = inRaw.bindMemory(to: UInt8.self)
                    for i in 0..<count {
                        out[i] = UInt16(bytes[i])
                    }
                }
            }
            return (promoted, count, .uint16)
        case 2:
            return (raw, count, .uint16)
        case 4:
            return (raw, count, .uint32)
        default:
            return nil
        }
    }

    /// Validate an index buffer against the mesh's vertex count and, if any
    /// triangle references a vertex `>= vertexCount`, return a filtered copy
    /// with those triangles dropped. Returns the input unchanged (cheap, no
    /// copy) when every index is in range — the overwhelmingly common case.
    /// Returns `nil` only if every triangle was out of range (nothing left to
    /// draw), so the caller skips the mesh rather than build an empty one.
    ///
    /// `from(scnGeometry:)` derives `vertexCount` from the `.vertex` source's
    /// `vectorCount` and the indices from the element independently; merged /
    /// imported / hand-built geometry can desync the two. Every downstream
    /// consumer indexes `positions[index]` trusting this invariant — enforce it
    /// once at the door so an out-of-range index can't SIGTRAP the render loop.
    static func sanitiseIndices(
        _ input: (data: Data, count: Int, type: MTLIndexType),
        vertexCount: Int,
        geometryLabel: String
    ) -> (data: Data, count: Int, type: MTLIndexType)? {
        let (data, count, type) = input
        // Decode to UInt32 so the bounds test is uniform across widths.
        var idx = [UInt32](); idx.reserveCapacity(count)
        data.withUnsafeBytes { raw in
            switch type {
            case .uint16:
                let p = raw.bindMemory(to: UInt16.self)
                for i in 0..<count { idx.append(UInt32(p[i])) }
            default:   // .uint32
                let p = raw.bindMemory(to: UInt32.self)
                for i in 0..<count { idx.append(p[i]) }
            }
        }
        let limit = UInt32(vertexCount)
        // Fast path: scan for any out-of-range index; if none, hand the
        // original buffer straight back (no allocation, no rebuild).
        guard idx.contains(where: { $0 >= limit }) else { return input }

        var kept = [UInt32](); kept.reserveCapacity(count)
        var t = 0
        while t + 2 < count {
            let a = idx[t], b = idx[t + 1], c = idx[t + 2]
            if a < limit && b < limit && c < limit { kept.append(a); kept.append(b); kept.append(c) }
            t += 3
        }
        let dropped = (count - kept.count) / 3
        log.notice("Illuminatorama mesh \(geometryLabel): \(dropped) of \(count / 3) triangles reference a vertex >= vertexCount \(vertexCount) (maxIndex \(idx.max() ?? 0)); dropped.")
        guard !kept.isEmpty else { return nil }
        // Re-emit as uint32 (always valid for the kept indices) so we don't
        // have to worry about whether the survivors still fit in uint16.
        let outData = kept.withUnsafeBytes { Data($0) }
        return (outData, kept.count, .uint32)
    }
}
