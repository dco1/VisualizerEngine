import simd

/// Hi-Z occlusion (Daydream DH-1466) — the CPU REFERENCE of the pyramid build (`SignackHiZ.metal`)
/// and of the per-cluster / per-instance test (`signack_occluded` in the cut and instance kernels).
///
/// The pyramid holds the FARTHEST depth over each texel's footprint (standard depth, 0 near / 1 far).
/// A sphere is hidden when the nearest corner of its bounding cube, projected, lies behind the max
/// depth over every texel its (dilated) projected footprint touches — at the level where that
/// footprint spans at most two texels. Anything reaching the near plane is visible.
public enum SignackOcclusion {

    /// One mip level: `width × height` depths, row-major.
    public struct Level: Sendable, Equatable {
        public var width: Int, height: Int
        public var depth: [Float]
        public init(width: Int, height: Int, depth: [Float]) { self.width = width; self.height = height; self.depth = depth }
        func at(_ x: Int, _ y: Int) -> Float { depth[min(y, height - 1) * width + min(x, width - 1)] }
    }

    /// The pyramid of a `width × height` depth buffer. Level 0 is half the size rounded up to a power
    /// of two per axis (texels past the source edge repeat it — they lie off screen), so every further
    /// level halves EXACTLY, down to 1 × 1, and level `L`'s texel spans 2^(L+1) source px.
    public static func pyramid(depth: [Float], width: Int, height: Int) -> [Level] {
        var src = Level(width: width, height: height, depth: depth)
        var out: [Level] = []
        var w = pow2((width + 1) / 2), h = pow2((height + 1) / 2)
        while true {
            var d = [Float](repeating: 0, count: w * h)
            for y in 0..<h {
                for x in 0..<w {
                    d[y * w + x] = max(max(src.at(2 * x, 2 * y), src.at(2 * x + 1, 2 * y)),
                                       max(src.at(2 * x, 2 * y + 1), src.at(2 * x + 1, 2 * y + 1)))
                }
            }
            src = Level(width: w, height: h, depth: d)
            out.append(src)
            if w == 1 && h == 1 { break }
            w = max(1, w / 2); h = max(1, h / 2)
        }
        return out
    }

    /// The smallest power of two ≥ `n` (≥ 1).
    public static func pow2(_ n: Int) -> Int { var p = 1; while p < n { p <<= 1 }; return p }

    /// Is sphere (`center`, `radius`) — in the space `toClip` maps from — hidden in `levels`
    /// (the pyramid of a `depthSize` depth), with each footprint grown by `dilate` px?
    public static func isOccluded(center c: SIMD3<Float>, radius r: Float, toClip m: simd_float4x4,
                                  levels: [Level], depthSize: SIMD2<Float>, dilate: Float) -> Bool {
        guard !levels.isEmpty else { return false }
        var lo = SIMD2<Float>(repeating: 1e30), hi = SIMD2<Float>(repeating: -1e30), zmin: Float = 1e30
        for k in 0..<8 {
            let p = c + r * SIMD3<Float>(k & 1 != 0 ? 1 : -1, k & 2 != 0 ? 1 : -1, k & 4 != 0 ? 1 : -1)
            let q = m * SIMD4(p, 1)
            if q.w <= 1e-4 { return false }
            let n = SIMD3(q.x, q.y, q.z) / q.w
            lo = simd_min(lo, SIMD2(n.x, n.y)); hi = simd_max(hi, SIMD2(n.x, n.y)); zmin = min(zmin, n.z)
        }
        if zmin <= 0 { return false }
        var pmin = SIMD2(lo.x * 0.5 + 0.5, 0.5 - hi.y * 0.5) * depthSize - dilate
        var pmax = SIMD2(hi.x * 0.5 + 0.5, 0.5 - lo.y * 0.5) * depthSize + dilate
        pmin = simd_clamp(pmin, .zero, depthSize - 1); pmax = simd_clamp(pmax, .zero, depthSize - 1)
        let ext = max(pmax.x - pmin.x, pmax.y - pmin.y)
        let L = Int(min(max(ceil(log2(max(ext, 1))) - 1, 0), Float(levels.count - 1)))
        let s = exp2(Float(L + 1))
        let lv = levels[L]
        let x1 = min(Int(pmax.x / s), lv.width - 1), y1 = min(Int(pmax.y / s), lv.height - 1)
        let x0 = min(Int(pmin.x / s), x1), y0 = min(Int(pmin.y / s), y1)
        var mx: Float = 0
        for y in y0...y1 { for x in x0...x1 { mx = max(mx, lv.at(x, y)) } }
        return zmin > mx
    }
}
