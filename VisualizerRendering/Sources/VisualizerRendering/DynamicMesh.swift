import Foundation
import Metal
import SceneKit

// ── DYNAMIC MESH ─────────────────────────────────────────────────────────────
//
// DynamicMesh backs an SCNGeometry's vertex positions with a .storageModeShared
// MTLBuffer. SceneKit reads the buffer on its render thread; a Metal compute
// kernel writes new positions into the same buffer just before SceneKit draws.
// No CPU round-trip, no copy, no SCNMorpher.
//
// SceneKit reads the MTLBuffer in-place each frame ("SceneKit uses this
// MTLBuffer directly as the source of vertex data, without copying the
// buffer's contents." — Apple SCNGeometrySource docs). This is the contract
// that makes the whole approach work.
//
// WHERE TO GO NEXT
// ─────────────────
// • Normals (Phase 2): the tube-expand kernel already has the per-vertex
//   radial direction, which IS the outward normal. Write it into normalBuffer
//   in the same pass. Wire normalBuffer into a second SCNGeometrySource with
//   semantic .normal. Lighting will dramatically improve for curved surfaces.
//
// • UVs (Phase 2): add a uvBuffer (SIMD2<Float>), generate arc-length UVs in
//   the tube-expand kernel (u = ring angle / 2π, v = arc-length / total-length).
//   Stable UVs allow tiling a sausage-casing texture without stretch.
//
// • Bone skinning alternative (Phase 3): for simpler chain counts (<8 segments),
//   drive SCNNode.position for pre-built capsule segments from PBD spine
//   positions read back on the CPU. SceneKit updates normals automatically.
//   Switch to DynamicMesh when your vertex count or frame-rate requirements
//   demand zero CPU overhead.
//
// • Variable-count geometry (Phase 5 — fluid/marching cubes): SCNGeometry has
//   a fixed index count. When the surface triangle count changes every frame
//   (marching cubes, metaballs), move the render pass entirely to Metal and
//   composite into SceneKit via an SCNTechnique. DynamicMesh is not the right
//   abstraction there.

@MainActor
public final class DynamicMesh {

    // The geometry to assign to your SCNNode.
    // SceneKit reads positionBuffer.buffer on its render thread each frame.
    public let geometry: SCNGeometry

    // GPU-backed position buffer. Write new positions here every frame from
    // the tube-expand (or any deformation) kernel.
    public let positionBuffer: SimBuffer<SIMD3<Float>>

    // GPU-backed normal buffer. Written each frame by the tube-expand kernel
    // as radial outward normals (packed_float3, 12-byte stride).
    public let normalBuffer: SimBuffer<SIMD3<Float>>

    // GPU-backed per-vertex color multiplier (SIMD4<Float>, stride 16).
    // Written each frame by pbdTubeExpand as a belly-curve tip-darkening ramp.
    // Initialized to identity white at allocation so non-PBD DynamicMesh
    // callers (Forest bark, HotdogDropPlus, etc.) see no color modulation —
    // pbdTubeExpand is never dispatched for those meshes.
    public let colorBuffer: SimBuffer<SIMD4<Float>>

    public let vertexCount: Int

    // ── Init ─────────────────────────────────────────────────────────────────

    // Indices live in shared MTLBuffers — topology is identical across every
    // instance of the same tube shape so we hand the same MTLBuffers to every
    // SCNGeometryElement. See `DynamicMeshIndexCache` below for the cache.
    //
    // Two elements: body (the cylindrical mid-section) and caps (the two
    // hemispherical end domes + pole fans). Each element gets its own material
    // slot so the renderer can paint a distinct cooked-cut tone on the caps
    // without affecting the casing.
    init?(device: MTLDevice,
          vertexCount: Int,
          bodyIndexBuffer: MTLBuffer,
          bodyIndexCount: Int,
          capIndexBuffer: MTLBuffer?,
          capIndexCount: Int) {
        guard
            let posBuf  = SimBuffer<SIMD3<Float>>(device: device,
                                                  capacity: vertexCount,
                                                  label: "DynamicMesh.positions"),
            let normBuf = SimBuffer<SIMD3<Float>>(device: device,
                                                  capacity: vertexCount,
                                                  label: "DynamicMesh.normals"),
            let colorBuf = SimBuffer<SIMD4<Float>>(device: device,
                                                   capacity: vertexCount,
                                                   label: "DynamicMesh.colors")
        else { return nil }

        // Seed the color buffer with identity white so non-pbdTubeExpand callers
        // (Forest bark, grid meshes, etc.) get no albedo modulation. Illuminatorama
        // multiplies albedo × color per vertex; white = no-op.
        let colorInitPtr = colorBuf.buffer.contents()
            .bindMemory(to: SIMD4<Float>.self, capacity: vertexCount)
        for i in 0..<vertexCount { colorInitPtr[i] = SIMD4<Float>(1, 1, 1, 1) }

        // Both position and normal buffers are written by the tube-expand kernel
        // as packed_float3 (12 bytes). MTLBuffer shared memory is zero-initialized;
        // the GPU overwrites before the first render tick.
        let stride12 = MemoryLayout<Float>.stride * 3  // 12 — matches packed_float3
        let posSource = SCNGeometrySource(
            buffer: posBuf.buffer,
            vertexFormat: .float3,
            semantic: .vertex,
            vertexCount: vertexCount,
            dataOffset: 0,
            dataStride: stride12
        )
        let normSource = SCNGeometrySource(
            buffer: normBuf.buffer,
            vertexFormat: .float3,
            semantic: .normal,
            vertexCount: vertexCount,
            dataOffset: 0,
            dataStride: stride12
        )

        // SceneKit treats the index buffer as read-only — passing the same
        // MTLBuffer to N SCNGeometryElement instances is supported and is the
        // whole point of the shared index cache.
        let bodyElement = SCNGeometryElement(
            buffer: bodyIndexBuffer,
            primitiveType: .triangles,
            primitiveCount: bodyIndexCount / 3,
            bytesPerIndex: MemoryLayout<UInt16>.stride
        )
        // Cap element is optional: tubes pass a non-nil cap buffer; single-element
        // meshes (e.g. `makeGrid`) pass capIndexCount == 0 → body-only geometry.
        var elements = [bodyElement]
        if let capBuf = capIndexBuffer, capIndexCount > 0 {
            elements.append(SCNGeometryElement(
                buffer: capBuf,
                primitiveType: .triangles,
                primitiveCount: capIndexCount / 3,
                bytesPerIndex: MemoryLayout<UInt16>.stride
            ))
        }
        // GPU-normal-fed mesh — normals come from `normalBuffer` (written by the
        // deform kernel / caller), not derived from winding.
        self.geometry       = SCNGeometry(sources: [posSource, normSource], // winding-ok: GPU-normal-fed; normals from normalBuffer, not winding
                                          elements: elements)
        self.positionBuffer = posBuf
        self.normalBuffer   = normBuf
        self.colorBuffer    = colorBuf
        self.vertexCount    = vertexCount

        // Phase 4.13a — publish a GPU mesh descriptor on the geometry so
        // Illuminatorama's scene extractor can build an interleaved
        // `IlluminatoramaVertex` buffer + repack task on first encounter.
        // The extractor sees this and skips the (empty) CPU-readback
        // path that previously made every DynamicMesh-backed asset
        // invisible through the overlay.
        //
        // colorBuffer is always non-nil (seeded white above). Non-PBD callers
        // (Forest bark, grid meshes) leave it white → identity multiply.
        // pbdTubeExpand writes the tip-darkening ramp each frame.
        self.geometry.illuminatoramaGPUMesh = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: posBuf.buffer,
            normalBuffer: normBuf.buffer,
            positionStride: stride12,
            normalStride: stride12,
            vertexCount: vertexCount,
            bodyIndexBuffer: bodyIndexBuffer,
            bodyIndexCount: bodyIndexCount,
            bodyIndexType: .uint16,
            capIndexBuffer: (capIndexCount > 0) ? capIndexBuffer : nil,
            capIndexCount: capIndexCount,
            capIndexType: .uint16,
            colorBuffer: colorBuf.buffer,
            colorStride: 16
        )
    }

    // ── Packed CPU vertex access (stride contract) ────────────────────────────
    //
    // `positionBuffer` / `normalBuffer` are typed `SimBuffer<SIMD3<Float>>` only
    // for allocation convenience — their *consumed* layout is `packed_float3`
    // (stride 12: 3 contiguous floats per vertex), the format both the
    // SCNGeometrySource (`dataStride: 12`) and `illumi_repack_pos_norm`
    // (`packed_float3*`) read. A CPU writer/reader that binds the raw buffer to
    // `SIMD3<Float>` (stride 16) shears every vertex after the first — the
    // documented SIMD3 stride trap. These accessors enforce the packed stride so
    // host-side deformers (e.g. a grid ripple) and the deforming-surface cache
    // re-frame agree byte-for-byte with what the GPU traces.

    /// Write one vertex's position + normal as packed `float3` (stride 12).
    public func writeGridVertex(_ i: Int, position: SIMD3<Float>, normal: SIMD3<Float>) {
        precondition(i >= 0 && i < vertexCount, "DynamicMesh vertex index out of range")
        let p = positionBuffer.buffer.contents().bindMemory(to: Float.self, capacity: vertexCount * 3)
        let n = normalBuffer.buffer.contents().bindMemory(to: Float.self, capacity: vertexCount * 3)
        p[3*i] = position.x; p[3*i+1] = position.y; p[3*i+2] = position.z
        n[3*i] = normal.x;   n[3*i+1] = normal.y;   n[3*i+2] = normal.z
    }

    // ── Tube factory ─────────────────────────────────────────────────────────

    // Build a tube mesh with `ringCount` rings, each ring having `ringSegments`
    // vertices. Rings are connected by quad strips; two flat end-cap fans close
    // the ends. `ringCount` is the number of visual rings written by the kernel —
    // for a Catmull-Rom oversampled tube it is (spineCount-1)*subSegments+1, not
    // the raw spine particle count.
    //
    // The index buffer is identical across every tube of the same shape, so we
    // pull it from a shared cache rather than re-building (and re-uploading) on
    // every spawn.
    public static func makeTube(
        device: MTLDevice,
        ringCount: Int,
        ringSegments: Int
    ) -> DynamicMesh? {
        // +2 for the two end-cap pole vertices written by the kernel.
        let tubeVerts   = ringCount * ringSegments
        let vertexCount = tubeVerts + 2
        guard vertexCount <= 65535 else { return nil }

        guard let cached = DynamicMeshIndexCache.shared
                .tubeIndices(device: device,
                             ringCount: ringCount,
                             ringSegments: ringSegments)
        else { return nil }

        return DynamicMesh(device: device,
                           vertexCount: vertexCount,
                           bodyIndexBuffer: cached.bodyBuffer,
                           bodyIndexCount:  cached.bodyIndexCount,
                           capIndexBuffer:  cached.capBuffer,
                           capIndexCount:   cached.capIndexCount)
    }

    // ── Grid factory (general deforming surface) ─────────────────────────────
    //
    // A flat `cols × rows` vertex grid spanning `width` (x) × `depth` (z), centred
    // on the local origin at y = 0, as a single triangle-list element (no caps).
    // Unlike `makeTube` (whose vertex layout is defined by the tube-expand kernel),
    // the grid's layout is a documented contract for external callers: vertex
    // (column `c`, row `r`) is at index `r * cols + c`. The caller owns the
    // deformation — write `positionBuffer` / `normalBuffer` each frame. Topology
    // (vertex + index count) is fixed for the mesh's life, so it rides the Phase-B
    // BLAS-refit path and (future) deforming surface-cache path. Seeded flat with
    // up-normals so it renders correctly before the first deformation write.
    public static func makeGrid(
        device: MTLDevice,
        cols: Int,
        rows: Int,
        width: Float,
        depth: Float
    ) -> DynamicMesh? {
        guard cols >= 2, rows >= 2, cols * rows <= 65535 else { return nil }
        let vertexCount = cols * rows

        // Two triangles per cell. Winding is CW-front (SceneKit's convention for
        // custom geometry) for an upward-facing grid; normals come from the buffer.
        var idx = [UInt16]()
        idx.reserveCapacity((cols - 1) * (rows - 1) * 6)
        for r in 0..<(rows - 1) {
            for c in 0..<(cols - 1) {
                let i00 = UInt16(r * cols + c)
                let i10 = UInt16(r * cols + c + 1)
                let i01 = UInt16((r + 1) * cols + c)
                let i11 = UInt16((r + 1) * cols + c + 1)
                idx.append(contentsOf: [i00, i01, i10,  i10, i01, i11])
            }
        }
        guard let indexBuffer = device.makeBuffer(
                bytes: idx,
                length: idx.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared) else { return nil }
        indexBuffer.label = "DynamicMesh.grid.indices"

        guard let mesh = DynamicMesh(
                device: device, vertexCount: vertexCount,
                bodyIndexBuffer: indexBuffer, bodyIndexCount: idx.count,
                capIndexBuffer: nil, capIndexCount: 0) else { return nil }

        // Seed a flat grid + up normals so the mesh is valid before the first
        // deformation write (the buffers are shared, so a direct CPU seed is fine).
        // STRIDE CONTRACT: these buffers are consumed as `packed_float3` (stride
        // 12) by BOTH the SCNGeometrySource (`dataStride: 12`) and the
        // `illumi_repack_pos_norm` kernel (`packed_float3*`). A CPU writer MUST
        // therefore write packed (3 contiguous floats / vertex) — binding to
        // `SIMD3<Float>` (stride 16) shears every vertex after the first (the
        // documented SIMD3 stride trap). See `writeGridVertex`.
        for r in 0..<rows {
            for c in 0..<cols {
                let u = Float(c) / Float(cols - 1)
                let v = Float(r) / Float(rows - 1)
                mesh.writeGridVertex(r * cols + c,
                                     position: SIMD3<Float>((u - 0.5) * width, 0, (v - 0.5) * depth),
                                     normal: SIMD3<Float>(0, 1, 0))
            }
        }
        return mesh
    }
}

// ── DYNAMIC MESH INDEX CACHE ─────────────────────────────────────────────────
//
// Every tube of a given (ringCount, ringSegments) shape has byte-for-byte the
// same index list. Building the list per spawn (one fresh `[UInt16]` of
// ~1500 entries) and uploading it to a new MTLBuffer was real CPU work at
// a 4/sec spawn rate. This cache builds the buffer once per shape per device
// and hands the same MTLBuffer to every DynamicMesh — SceneKit reads it
// concurrently from many SCNGeometryElement instances without conflict.
@MainActor
final class DynamicMeshIndexCache {

    static let shared = DynamicMeshIndexCache()
    private init() {}

    struct Entry {
        let bodyBuffer: MTLBuffer
        let bodyIndexCount: Int
        let capBuffer: MTLBuffer
        let capIndexCount: Int
    }

    private struct Key: Hashable {
        let deviceID: ObjectIdentifier
        let ringCount: Int
        let ringSegments: Int
    }

    private var entries: [Key: Entry] = [:]

    func tubeIndices(device: MTLDevice, ringCount: Int, ringSegments: Int) -> Entry? {
        let key = Key(deviceID: ObjectIdentifier(device),
                      ringCount: ringCount, ringSegments: ringSegments)
        if let e = entries[key] { return e }

        let tubeVerts = ringCount * ringSegments

        // Cap / body split mirrors the region split in PBD.metal's pbdTubeExpand:
        //   K cap rings at each end, body rings in the middle. With ringCount ≥ 5
        //   we get K = min(3, (ringCount-2)/3); for a 4-particle / 13-ring tube
        //   that's K = 3 — first 3 and last 3 rings are hemispherical cap rings.
        // Putting the rim quad-strip (ring K-1 ↔ ring K) in the CAP element
        // extends the cooked-cut tone right up to the visual dome boundary.
        let K = max(1, min(3, (ringCount - 2) / 3))
        let bodyStart = K               // first body ring
        let bodyEnd   = ringCount - K   // first end-cap ring

        var bodyIdx = [UInt16]()
        var capIdx  = [UInt16]()
        bodyIdx.reserveCapacity((bodyEnd - bodyStart - 1) * ringSegments * 6)
        capIdx.reserveCapacity(K * 2 * ringSegments * 6 + ringSegments * 6)

        // Body quad strips: between body rings only (K…bodyEnd-1).
        for s in bodyStart..<(bodyEnd - 1) {
            for r in 0..<ringSegments {
                let next = (r + 1) % ringSegments
                let a = UInt16(s * ringSegments + r)
                let b = UInt16(s * ringSegments + next)
                let c = UInt16((s + 1) * ringSegments + r)
                let d = UInt16((s + 1) * ringSegments + next)
                bodyIdx.append(contentsOf: [a, b, c, b, d, c])
            }
        }

        // Cap quad strips: every strip on either end, including the rim
        // strip (K-1 ↔ K) and the end-rim strip (bodyEnd-1 ↔ bodyEnd).
        let capStripRanges: [Range<Int>] = [
            0..<bodyStart,                  // start cap interior + rim
            (bodyEnd - 1)..<(ringCount - 1) // end cap rim + interior
        ]
        for range in capStripRanges {
            for s in range {
                for r in 0..<ringSegments {
                    let next = (r + 1) % ringSegments
                    let a = UInt16(s * ringSegments + r)
                    let b = UInt16(s * ringSegments + next)
                    let c = UInt16((s + 1) * ringSegments + r)
                    let d = UInt16((s + 1) * ringSegments + next)
                    capIdx.append(contentsOf: [a, b, c, b, d, c])
                }
            }
        }

        // Start cap fan: pole at tubeVerts, outward normal = −tangent.
        let cap0 = UInt16(tubeVerts)
        for r in 0..<ringSegments {
            capIdx.append(contentsOf: [cap0, UInt16((r + 1) % ringSegments), UInt16(r)])
        }
        // End cap fan: pole at tubeVerts+1, outward normal = +tangent.
        let cap1 = UInt16(tubeVerts + 1)
        let lastBase = (ringCount - 1) * ringSegments
        for r in 0..<ringSegments {
            capIdx.append(contentsOf: [cap1, UInt16(lastBase + r),
                                       UInt16(lastBase + (r + 1) % ringSegments)])
        }

        func upload(_ indices: [UInt16], label: String) -> MTLBuffer? {
            let byteLen = MemoryLayout<UInt16>.stride * indices.count
            guard byteLen > 0 else { return nil }
            let buf = indices.withUnsafeBufferPointer { ptr -> MTLBuffer? in
                device.makeBuffer(bytes: ptr.baseAddress!,
                                  length: byteLen,
                                  options: .storageModeShared)
            }
            buf?.label = label
            return buf
        }

        guard let bodyBuf = upload(bodyIdx, label: "DynamicMesh.body[\(ringCount)x\(ringSegments)]"),
              let capBuf  = upload(capIdx,  label: "DynamicMesh.caps[\(ringCount)x\(ringSegments)]")
        else { return nil }

        let entry = Entry(bodyBuffer: bodyBuf,    bodyIndexCount: bodyIdx.count,
                          capBuffer:  capBuf,     capIndexCount:  capIdx.count)
        entries[key] = entry
        return entry
    }
}
