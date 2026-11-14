import Metal
import simd

// THE RT NORMAL/UV SLAB (Daydream DH-1672)
// ========================================
//
// A TLAS hit resolves its surface through `objNormal[normalBase + primitive_id]` and its mesh UVs
// through `objUV[(normalBase + primitive_id) * 3 …]`: one slice per BLAS'd mesh in two shared
// buffers. Every rebuild used to size and fill both from scratch — on 4000 Sunset ~400 meshes'
// worth, 5 ms of a wall edit's ~19 ms rebuild, although a swap rebuild re-points a few instances at a
// handful of new meshes and every other slice is byte-for-byte what it was.
//
// The buffers are now a slab that outlives the rebuild. A mesh keeps its slice for as long as its
// BLAS is cached; a rebuild writes only the meshes the slab has not seen, after everything already
// written. Appending never touches a byte an in-flight frame can read (that frame's instance rows
// only name slices written before it), and a compaction — when the slab is full or more than half
// holes — writes a fresh pair of buffers, leaving the old pair to the frames that hold it.
extension IlluminatoramaRenderer {

    /// Give every entry a slice and return each mesh's `normalBase`. nil when allocation failed.
    func placeRTNormalSlices(_ entries: [RTMeshEntry]) -> [ObjectIdentifier: Int]? {
        // Slices of meshes whose BLAS left the cache are holes now (the cache holds its meshes strongly,
        // so an identifier still in it cannot have been reused by a new mesh).
        if !rts.rtSlabBase.isEmpty { rts.rtSlabBase = rts.rtSlabBase.filter { rts.rtMeshCache[$0.key] != nil } }
        let live = entries.reduce(0) { $0 + $1.normals.count }
        let fresh = entries.filter { rts.rtSlabBase[ObjectIdentifier($0.mesh)] == nil }
        let need = fresh.reduce(0) { $0 + $1.normals.count }
        let compact = !rts.rtNormalSlabEnabled || rts.rtObjNormalBuffer == nil || rts.rtObjUVBuffer == nil
            || rts.rtSlabUsed + need > rts.rtSlabCapacity
            || rts.rtSlabUsed - live > max(live, 4096)
        if compact {
            let cap = rts.rtNormalSlabEnabled ? max(live + live / 2, 4096) : live
            guard cap > 0,
                  let nbuf = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * cap, options: .storageModeShared),
                  let ubuf = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * cap * 3, options: .storageModeShared)
            else { return nil }
            nbuf.label = "Illuminatorama.rt.objNormal"
            ubuf.label = "Illuminatorama.rt.objUV"
            rts.rtObjNormalBuffer = nbuf; rts.rtObjUVBuffer = ubuf
            rts.rtSlabCapacity = cap; rts.rtSlabUsed = 0; rts.rtSlabBase = [:]
            writeRTNormalSlices(entries)
        } else {
            writeRTNormalSlices(fresh)
        }
        rts.rtObjNormalCount = rts.rtSlabUsed
        rts.rtObjUVCount = rts.rtSlabUsed * 3
        var out: [ObjectIdentifier: Int] = [:]
        for e in entries { out[ObjectIdentifier(e.mesh)] = rts.rtSlabBase[ObjectIdentifier(e.mesh)] ?? 0 }
        return out
    }

    /// Append `entries`' normals and UVs at the slab's end. A mesh whose UVs could not be read
    /// (GPU-private vertices) is padded with zeros, so every slice keeps 3 UVs per normal.
    private func writeRTNormalSlices(_ entries: [RTMeshEntry]) {
        guard let nbuf = rts.rtObjNormalBuffer, let ubuf = rts.rtObjUVBuffer else { return }
        let np = nbuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: rts.rtSlabCapacity)
        let up = ubuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: rts.rtSlabCapacity * 3)
        var n = rts.rtSlabUsed
        for e in entries {
            let mid = ObjectIdentifier(e.mesh)
            guard rts.rtSlabBase[mid] == nil else { continue }
            rts.rtSlabBase[mid] = n
            e.normals.withUnsafeBufferPointer { src in
                if let b = src.baseAddress { (np + n).update(from: b, count: src.count) }
            }
            let want = e.normals.count * 3
            let have = min(want, e.uvs.count)
            e.uvs.withUnsafeBufferPointer { src in
                if have > 0, let b = src.baseAddress { (up + n * 3).update(from: b, count: have) }
            }
            if have < want { (up + n * 3 + have).update(repeating: .zero, count: want - have) }
            n += e.normals.count
        }
        rts.rtSlabUsed = n
    }
}
