//  MeshIO+GPUSTL.swift — binary STL on the GPU: decode (optionally welded + smoothed in the same command buffer) and
//  encode. See MeshIO+GPU.swift for what is guaranteed and when the CPU path runs instead.

import Foundation

#if canImport(Metal)
import Metal

extension MeshIO {
    /// Binary STL -> floats (+ weld and smooth when asked). nil = not a case for the GPU (ascii, no device, 0 facets).
    static func gpuDecodeSTL(_ data: Data, options: ReadOptions) -> MeshIOFloatMesh? {
        guard let g = MeshIOGPU.shared, isBinarySTL(data) else { return nil }
        let count = data.withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 80, as: UInt32.self))) }
        guard count > 0, count < (1 << 29) / 3 else { return nil }
        let raw = g.upload(data), nV = 3 * count
        let pos = g.alloc(nV * 12), nrm = g.alloc(nV * 12)
        var out = MeshIOFloatMesh()
        if !options.weldAndSmooth {
            guard g.run({ e in g.go(e, "stl_decode", count, [.buf(raw), .buf(pos), .buf(nrm), .u(UInt32(count))]) }) else { return nil }
            out.positions = MeshIOGPU.read(pos, Float.self, count: nV * 3)
            out.normals = MeshIOGPU.read(nrm, Float.self, count: nV * 3)
            out.indices = (0 ..< UInt32(nV)).map { $0 }
            return out
        }
        var welded: MeshIOGPU.Welded?
        guard g.run({ e in
            g.go(e, "stl_decode", count, [.buf(raw), .buf(pos), .buf(nrm), .u(UInt32(count))])
            welded = g.encodeWeldSmooth(e, P: pos, nV: nV, I: nil, nT: count, crease: options.creaseDegrees, quantum: nil)
        }), let w = welded else { return nil }
        let c = MeshIOGPU.read(w.cnt, UInt32.self, count: 2)
        let kv = Int(c[0]), kt = Int(c[1])
        guard kt > 0 else { return nil }
        out.positions = MeshIOGPU.read(w.positions, Float.self, count: kv * 3)
        out.normals = MeshIOGPU.read(w.normals, Float.self, count: kv * 3)
        out.indices = MeshIOGPU.read(w.indices, UInt32.self, count: kt * 3)
        return out
    }

    static func gpuReadSTL(_ data: Data, options: ReadOptions) -> Result? {
        guard let f = gpuDecodeSTL(data, options: options) else { return nil }
        var r = Result()
        r.meshes = [f.toMesh3()]; r.names = [""]; r.materialNames = [nil]; r.colors = [[]]
        return r
    }

    /// Binary STL bytes for the meshes merged into one: header and count on the CPU, every 50-byte facet on the GPU,
    /// written straight into a page-aligned allocation the GPU maps (no copy back).
    static func gpuWriteSTL(_ meshes: [Mesh3]) -> Data? {
        guard let g = MeshIOGPU.shared else { return nil }
        var all = Mesh3()
        if meshes.count == 1 { all = meshes[0] } else { for m in meshes { all.append(m) } }
        let nT = all.indices.count / 3, nV = all.positions.count
        guard nT > 0, nT < (1 << 28), all.indices.count % 3 == 0 else { return nil }
        let total = 84 + 50 * nT
        let page = Int(getpagesize()), cap = (total + page - 1) / page * page
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, page, cap) == 0, let mem = raw else { return nil }
        memset(mem, 0, 84)
        let tag = Array("VisualizerMaterials MeshIO binary STL".utf8)
        tag.withUnsafeBytes { mem.copyMemory(from: $0.baseAddress!, byteCount: tag.count) }
        mem.storeBytes(of: UInt32(nT).littleEndian, toByteOffset: 80, as: UInt32.self)
        guard let out = g.device.makeBuffer(bytesNoCopy: mem, length: cap, options: .storageModeShared, deallocator: nil) else {
            free(mem); return nil
        }
        let P = g.upload(MeshIOGPUUtil.narrow3(all.positions)), I = g.upload(all.indices)
        guard g.run({ e in g.go(e, "stl_encode", nT, [.buf(P), .buf(I), .buf(out), .u(UInt32(nT)), .u(UInt32(nV))]) }) else {
            free(mem); return nil
        }
        return Data(bytesNoCopy: mem, count: total, deallocator: .free)
    }
}
#endif
