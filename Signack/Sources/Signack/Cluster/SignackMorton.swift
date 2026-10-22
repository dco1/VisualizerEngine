import simd

/// Z-order (Morton) sorting — the spatial-coherence fallback for geometry with no shared edges
/// (leaf cards) and the seed order for greedy clustering, so clusters are found in a stable,
/// spatially sweeping order.
enum SignackMorton {
    /// Spread the low 10 bits of `v` so there are two zero bits between each.
    @inline(__always) static func spread(_ v: UInt32) -> UInt32 {
        var x = v & 0x3FF
        x = (x | (x << 16)) & 0x0300_00FF
        x = (x | (x << 8)) & 0x0300_F00F
        x = (x | (x << 4)) & 0x030C_30C3
        x = (x | (x << 2)) & 0x0924_9249
        return x
    }

    /// Indices of `points`, sorted by 30-bit Morton code over their own bounding box (ties by index,
    /// so the order is total and deterministic).
    static func order(_ points: [SIMD3<Float>]) -> [Int] {
        guard !points.isEmpty else { return [] }
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
        for p in points { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let ext = simd_max(hi - lo, SIMD3(repeating: 1e-9))
        let codes: [UInt32] = points.map { p in
            let q = simd_clamp((p - lo) / ext, SIMD3(repeating: 0), SIMD3(repeating: 1)) * 1023
            return spread(UInt32(q.x)) | (spread(UInt32(q.y)) << 1) | (spread(UInt32(q.z)) << 2)
        }
        return points.indices.sorted { codes[$0] != codes[$1] ? codes[$0] < codes[$1] : $0 < $1 }
    }
}
