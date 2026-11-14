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
//
// SEVERAL RUNS (Daydream DH-1680). A merged group whose triangle COUNT held — the hot half of a
// drag's opening trim, a room's baseboards, where every moved wall changes its own run in place —
// can change two runs far apart in the list (the two walls at the dragged junction), and the one
// prefix/suffix middle then spans everything between: a 7 m box that covered every window-portal
// slice whole. With equal counts, triangle t of one mesh stands where triangle t of the other
// stood, so the pairs that differ hold the whole multiset difference: one box per differing RUN is
// exact, and costs the same single pass over the middle. Runs past `maxRuns` merge with their
// index neighbour (smallest union first). A count that changed keeps the single middle.
extension IlluminatoramaMesh {

    enum ShapeDiff: Equatable {
        /// Same triangles, same raster flags — the swap changed no depth anywhere.
        case identical
        /// Object-space bounds of every triangle that is not common to both.
        case region(lo: SIMD3<Float>, hi: SIMD3<Float>)
        /// Equal triangle counts, the differing pairs in several runs: one object-space box per run
        /// (≥ 2 boxes; their union is what `.region` would have said).
        case regions([Box])
        /// Cannot be compared (GPU-only buffers, different raster flags, too large).
        case whole

        struct Box: Equatable { var lo: SIMD3<Float>; var hi: SIMD3<Float> }
    }

    /// Past this many triangles (both meshes together) the comparison is not attempted.
    static let shapeDiffTriangleCap = 2_000_000

    /// Past this many differing runs the per-run boxes are not worth keeping: one middle box.
    static let shapeDiffRunCap = 512

    /// `maxRuns` > 1 lets an equal-count swap answer with up to that many run boxes (DH-1680).
    @MainActor static func shapeDiff(_ a: IlluminatoramaMesh, _ b: IlluminatoramaMesh,
                                     maxRuns: Int = 1) -> ShapeDiff {
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
        if maxRuns > 1, na == nb, let runs = differingRuns(ta, tb, prefix ..< (na - suffix), maxRuns: maxRuns) {
            if runs.isEmpty { return .identical }
            if runs.count > 1 { return .regions(runs) }
            return .region(lo: runs[0].lo, hi: runs[0].hi)
        }
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

    /// Equal-count meshes: one box per maximal run of index-aligned triangle pairs that differ (both
    /// poses), merged down to `maxRuns`. nil = a non-finite position or too many runs (the caller
    /// keeps the single middle box).
    private static func differingRuns(_ ta: TriangleReader, _ tb: TriangleReader, _ range: Range<Int>,
                                      maxRuns: Int) -> [ShapeDiff.Box]? {
        var runs: [ShapeDiff.Box] = []
        var open = false
        for t in range {
            let x = ta.triangle(t), y = tb.triangle(t)
            guard x != y else { open = false; continue }
            let lo = simd_min(simd_min(x.0, simd_min(x.1, x.2)), simd_min(y.0, simd_min(y.1, y.2)))
            let hi = simd_max(simd_max(x.0, simd_max(x.1, x.2)), simd_max(y.0, simd_max(y.1, y.2)))
            guard lo.x.isFinite, lo.y.isFinite, lo.z.isFinite, hi.x.isFinite, hi.y.isFinite, hi.z.isFinite else { return nil }
            if open {
                runs[runs.count - 1].lo = simd_min(runs[runs.count - 1].lo, lo)
                runs[runs.count - 1].hi = simd_max(runs[runs.count - 1].hi, hi)
            } else {
                if runs.count >= shapeDiffRunCap { return nil }
                runs.append(.init(lo: lo, hi: hi)); open = true
            }
        }
        // Merge index neighbours, the pair whose union is smallest first, down to `maxRuns`.
        func diag(_ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> Float { simd_length(hi - lo) }
        while runs.count > max(1, maxRuns) {
            var best = 0, bestCost = Float.greatestFiniteMagnitude
            for i in 0 ..< runs.count - 1 {
                let c = diag(simd_min(runs[i].lo, runs[i + 1].lo), simd_max(runs[i].hi, runs[i + 1].hi))
                if c < bestCost { bestCost = c; best = i }
            }
            runs[best].lo = simd_min(runs[best].lo, runs[best + 1].lo)
            runs[best].hi = simd_max(runs[best].hi, runs[best + 1].hi)
            runs.remove(at: best + 1)
        }
        return runs
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
