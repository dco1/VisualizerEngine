import simd

/// A Garland–Heckbert error quadric — the sum of squared distances to a set of planes — kept
/// area-weighted with its total weight, so `error(at:)` is a mean squared distance (m²) whose
/// square root is an RMS distance in object units.
struct SignackQuadric {
    // Symmetric 4×4 in upper-triangular order: a² ab ac ad b² bc bd c² cd d².
    var m = [Double](repeating: 0, count: 10)
    var weight: Double = 0

    /// The quadric of the plane through a triangle, weighted by its area.
    init(a: SIMD3<Float>, b: SIMD3<Float>, c: SIMD3<Float>) {
        let cr = simd_cross(SIMD3<Double>(b - a), SIMD3<Double>(c - a))
        let len = simd_length(cr)
        guard len > 1e-18 else { return }
        let n = cr / len
        let d = -simd_dot(n, SIMD3<Double>(a))
        let w = len * 0.5
        m = [n.x * n.x, n.x * n.y, n.x * n.z, n.x * d,
             n.y * n.y, n.y * n.z, n.y * d,
             n.z * n.z, n.z * d, d * d].map { $0 * w }
        weight = w
    }

    init() {}

    static func += (l: inout SignackQuadric, r: SignackQuadric) {
        for i in 0..<10 { l.m[i] += r.m[i] }
        l.weight += r.weight
    }

    static func + (l: SignackQuadric, r: SignackQuadric) -> SignackQuadric {
        var q = l; q += r; return q
    }

    /// Weighted mean squared distance of `p` to the quadric's planes (≥ 0).
    func error(at p: SIMD3<Float>) -> Double {
        let x = Double(p.x), y = Double(p.y), z = Double(p.z)
        let e = m[0] * x * x + 2 * m[1] * x * y + 2 * m[2] * x * z + 2 * m[3] * x
            + m[4] * y * y + 2 * m[5] * y * z + 2 * m[6] * y
            + m[7] * z * z + 2 * m[8] * z + m[9]
        return weight > 0 ? max(0, e / weight) : 0
    }
}
