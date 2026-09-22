import Foundation
import simd

// Transform, interpolation and noise helpers for the tree bake.
extension ForestTreeGeometry {

    // ── Transform helpers ─────────────────────────────────────────────────

    public static func xf(_ m: simd_float4x4, _ p: SIMD3<Float>) -> SIMD3<Float> {
        let r = m * SIMD4<Float>(p, 1)
        return SIMD3(r.x, r.y, r.z)
    }
    public static func xfDir(_ m: simd_float4x4, _ d: SIMD3<Float>) -> SIMD3<Float> {
        let r = m * SIMD4<Float>(d, 0)
        return safeNormalize(SIMD3(r.x, r.y, r.z))
    }
    public static func translate(_ t: SIMD3<Float>) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4(t, 1)
        return m
    }
    public static func rotX(_ a: Float) -> simd_float4x4 {
        let c = cos(a), s = sin(a)
        return simd_float4x4(SIMD4(1, 0, 0, 0), SIMD4(0, c, s, 0),
                             SIMD4(0, -s, c, 0), SIMD4(0, 0, 0, 1))
    }
    public static func rotY(_ a: Float) -> simd_float4x4 {
        let c = cos(a), s = sin(a)
        return simd_float4x4(SIMD4(c, 0, -s, 0), SIMD4(0, 1, 0, 0),
                             SIMD4(s, 0, c, 0), SIMD4(0, 0, 0, 1))
    }
    /// Rotation mapping local +Y onto `target`.
    public static func orientYTo(_ target: SIMD3<Float>) -> simd_float4x4 {
        let t = simd_normalize(target)
        let q = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: t)
        return simd_float4x4(q)
    }

    /// Cubic-Hermite interpolation over (t, r) control points with averaged-slope
    /// tangents — shared by the trunk + branch radius profiles.
    public static func hermite(_ cps: [(t: Float, r: Float)], _ t: Float) -> Float {
        for i in 0..<(cps.count - 1) where t <= cps[i + 1].t {
            let p0 = cps[i], p1 = cps[i + 1]
            let span = p1.t - p0.t
            let lt = span > 0 ? (t - p0.t) / span : 0
            let m0: Float = i > 0
                ? (p1.r - cps[i - 1].r) / (p1.t - cps[i - 1].t) * span : (p1.r - p0.r)
            let m1: Float = i + 2 < cps.count
                ? (cps[i + 2].r - p0.r) / (cps[i + 2].t - p0.t) * span : (p1.r - p0.r)
            let t2 = lt * lt, t3 = t2 * lt
            return (2 * t3 - 3 * t2 + 1) * p0.r + (t3 - 2 * t2 + lt) * m0
                 + (-2 * t3 + 3 * t2) * p1.r + (t3 - t2) * m1
        }
        return cps.last!.r
    }

    // ── Noise / value-noise / interpolation support (was file-scope) ──────
    @inline(__always) public static func mixv(_ a: SIMD3<Float>, _ b: SIMD3<Float>,
                                         _ t: Float) -> SIMD3<Float> {
        a + (b - a) * max(0, min(1, t))
    }

    // ── Degenerate-safe normalize (NaN guard) ───────────────────────────────────
    // `simd_normalize(v)` returns NaN when `|v| ≈ 0` (v/0). That happens all over the
    // tree bake on the random draws that make a sliver leaf card, a cancelling
    // summed-normal, or an axis-aligned cross product — and a NaN vertex normal ships
    // invalid geometry (broken lighting, and it fails the vertex-validity gate). Every
    // normalize on a vector that CAN collapse must fall back to a sensible unit vector
    // instead of dividing by zero. `fallback` MUST itself be a finite unit vector.
    @inline(__always) public static func safeNormalize(_ v: SIMD3<Float>,
                                         fallback: SIMD3<Float> = SIMD3<Float>(0, 1, 0))
        -> SIMD3<Float> {
        let len = simd_length(v)
        // 1e-6 chosen well above f32 denormal noise; below it v/len underflows to
        // garbage/NaN. Also reject a non-finite input outright.
        guard len > 1e-6, v.x.isFinite, v.y.isFinite, v.z.isFinite else { return fallback }
        return v / len
    }

    @inline(__always) public static func smoothstepf(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - e0) / (e1 - e0 == 0 ? 1 : (e1 - e0))))
        return t * t * (3 - 2 * t)
    }

    /// sRGB display value → linear, per channel. The Illuminatorama soup treats
    /// per-vertex colour as linear albedo, so the field-guide LookProfile colours
    /// (authored as sRGB) must be decoded — sister of the extractor's solid-colour
    /// sRGB→linear decode that fixed the washed-out translated scenes.
    @inline(__always) public static func srgbToLinear(_ c: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(pow(c.x, 2.2), pow(c.y, 2.2), pow(c.z, 2.2))
    }

    /// Halton low-discrepancy sequence (radical inverse) — used to scatter
    /// canopy-gap decisions evenly (blue-noise-ish) rather than in random clumps,
    /// so a real crown's lacy, sky-holed silhouette emerges instead of a packed
    /// shell. `base` 2 or 3 for the two axes.
    @inline(__always) public static func halton(_ index: UInt32, _ base: UInt32) -> Float {
        var i = index &+ 1
        var f: Float = 1
        var r: Float = 0
        while i > 0 {
            f /= Float(base)
            r += f * Float(i % base)
            i /= base
        }
        return r
    }

    @inline(__always) public static func hash2(_ x: Int, _ y: Int) -> Float {
        var h = UInt32(bitPattern: Int32(truncatingIfNeeded: x &* 374761393 &+ y &* 668265263))
        h = (h ^ (h >> 13)) &* 1274126177
        h = h ^ (h >> 16)
        return Float(h) * (1.0 / Float(UInt32.max))
    }

    public static func valueNoise(_ x: Float, _ z: Float) -> Float {
        let xi = floor(x), zi = floor(z)
        let xf = x - xi, zf = z - zi
        let u = xf * xf * (3 - 2 * xf)
        let v = zf * zf * (3 - 2 * zf)
        let x0 = Int(xi), z0 = Int(zi)
        let a = hash2(x0, z0),     b = hash2(x0 + 1, z0)
        let c = hash2(x0, z0 + 1), d = hash2(x0 + 1, z0 + 1)
        let ab = a + (b - a) * u
        let cd = c + (d - c) * u
        return ab + (cd - ab) * v
    }

    public static func fbm(_ x: Float, _ z: Float, octaves: Int) -> Float {
        var sum: Float = 0, amp: Float = 0.5, freq: Float = 1, norm: Float = 0
        for _ in 0..<octaves {
            sum += amp * valueNoise(x * freq, z * freq)
            norm += amp
            amp *= 0.5
            freq *= 2.03
        }
        return sum / max(1e-5, norm)
    }
}
