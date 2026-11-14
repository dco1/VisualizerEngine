import Metal
import simd

// WHERE DID A SWAPPED MESH ACTUALLY CHANGE? (Daydream DH-1668)
// ============================================================
//
// A host that edits geometry swaps the instance's mesh for a new one (Daydream's structural load
// re-registers a merged group whenever any wall in it moves). The shadow reuse (DH-1590 spot
// slices, DH-1668 sun cascades) then had to dirty the WHOLE old and new mesh — and a merged group
// spans a floor, so one dragged junction dirtied every cone and every cascade in the house.
//
// A shadow map's depth at a texel is the minimum over the triangles that cover it, so it changes
// only where a triangle was added or removed. When both meshes are CPU-authored, this compares
// their triangle lists (as resolved positions) from the front and from the back; the triangles
// outside the common prefix and suffix contain the whole multiset difference, and their
// object-space bounds are the region the swap can have changed. Exact for append-ordered merges
// (a moved wall changes its own run in the middle); for anything else the middle grows toward the
// whole mesh, which is the old answer, never a wrong one.
//
// Measured on 4000 Sunset's wall drag (63 swapped meshes a frame): ~0.5 ms, and the dirtied boxes
// fell from 126 (largest 31.7 m) to 42 (largest 18.2 m). An order-free multiset pass over the
// middles shrank them further (largest 12.2 m) but cost 8 ms a frame — not kept.
extension IlluminatoramaMesh {

    enum ShapeDiff: Equatable {
        /// Same triangles, same raster flags — the swap changed no depth anywhere.
        case identical
        /// Object-space bounds of every triangle that is not common to both.
        case region(lo: SIMD3<Float>, hi: SIMD3<Float>)
        /// Cannot be compared (GPU-only buffers, different raster flags, too large).
        case whole
    }

    /// Past this many triangles (both meshes together) the comparison is not attempted.
    static let shapeDiffTriangleCap = 2_000_000

    @MainActor static func shapeDiff(_ a: IlluminatoramaMesh, _ b: IlluminatoramaMesh) -> ShapeDiff {
        if a === b { return .identical }
        guard a.cpuAuthoredVertices, b.cpuAuthoredVertices,
              a.vertexBuffer.storageMode != .private, b.vertexBuffer.storageMode != .private,
              a.indexBuffer.storageMode != .private, b.indexBuffer.storageMode != .private,
              a.indirectDrawBuffer == nil, b.indirectDrawBuffer == nil,
              a.gpuInstanceBuffer == nil, b.gpuInstanceBuffer == nil,
              a.leafCutout == b.leafCutout, !a.leafCutout,
              a.shadowCastsBothFaces == b.shadowCastsBothFaces, a.doubleSided == b.doubleSided,
              a.tracedOnly == b.tracedOnly else { return .whole }
        let na = a.indexCount / 3, nb = b.indexCount / 3
        guard na + nb <= shapeDiffTriangleCap else { return .whole }
        let ta = TriangleReader(a), tb = TriangleReader(b)
        let common = min(na, nb)
        var prefix = 0
        while prefix < common, ta.triangle(prefix) == tb.triangle(prefix) { prefix += 1 }
        var suffix = 0
        while suffix < common - prefix, ta.triangle(na - 1 - suffix) == tb.triangle(nb - 1 - suffix) { suffix += 1 }
        if prefix + suffix == na, na == nb { return .identical }
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var any = false
        for (r, n) in [(ta, na), (tb, nb)] {
            for t in prefix ..< (n - suffix) {
                let tri = r.triangle(t)
                let (p0, p1, p2) = tri
                lo = simd_min(lo, simd_min(p0, simd_min(p1, p2)))
                hi = simd_max(hi, simd_max(p0, simd_max(p1, p2)))
                any = true
            }
        }
        if !any { return .identical }
        guard lo.x.isFinite, lo.y.isFinite, lo.z.isFinite, hi.x.isFinite, hi.y.isFinite, hi.z.isFinite,
              lo.x <= hi.x else { return .whole }
        return .region(lo: lo, hi: hi)
    }

    /// Resolves triangle `t` of a CPU-visible mesh to its three object-space positions.
    private struct TriangleReader {
        let verts: UnsafeRawPointer, idx: UnsafeRawPointer
        let wide: Bool, vertexCount: Int
        static let stride = MemoryLayout<IlluminatoramaVertex>.stride

        @MainActor init(_ m: IlluminatoramaMesh) {
            verts = UnsafeRawPointer(m.vertexBuffer.contents())
            idx = UnsafeRawPointer(m.indexBuffer.contents())
            wide = m.indexType == .uint32
            vertexCount = m.vertexCount
        }

        @inline(__always) func index(_ i: Int) -> Int {
            wide ? Int(idx.load(fromByteOffset: i * 4, as: UInt32.self))
                 : Int(idx.load(fromByteOffset: i * 2, as: UInt16.self))
        }

        @inline(__always) func position(_ v: Int) -> SIMD3<Float> {
            // An out-of-range index is the mesh's own defect; read it as NaN so it never compares
            // equal (both sides then fall in the middle) and fails the finite check (→ whole).
            guard v < vertexCount else { return SIMD3(repeating: .nan) }
            return verts.load(fromByteOffset: v * Self.stride, as: SIMD3<Float>.self)
        }

        @inline(__always) func triangle(_ t: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
            (position(index(3 * t)), position(index(3 * t + 1)), position(index(3 * t + 2)))
        }
    }
}

extension IlluminatoramaCullBounds {
    /// World bounds of an object-space box under a rigid (or affine) model matrix, with the same
    /// float-error margin `worldSphere` carries. nil when the result is not finite.
    static func worldBox(lo: SIMD3<Float>, hi: SIMD3<Float>, model m: simd_float4x4) -> IlluminatoramaCullBounds? {
        var b = IlluminatoramaCullBounds.empty
        for i in 0..<8 {
            let p = SIMD3<Float>((i & 1) == 0 ? lo.x : hi.x, (i & 2) == 0 ? lo.y : hi.y, (i & 4) == 0 ? lo.z : hi.z)
            let w = m * SIMD4<Float>(p, 1)
            let q = SIMD3<Float>(w.x, w.y, w.z)
            b.lo = simd_min(b.lo, q); b.hi = simd_max(b.hi, q)
        }
        guard b.isBounded else { return nil }
        let margin = simd_length(b.hi - b.lo) * 0.0005 + 0.002
        b.lo -= SIMD3(repeating: margin); b.hi += SIMD3(repeating: margin)
        return b
    }
}
