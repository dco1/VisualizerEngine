//  Displace — push every vertex by a scalar field, along its normal or a fixed direction.
//
//      p' = p + dir · strength · (field(p, uv) − midlevel)
//
//  `field` is a closure (position, uv) -> value, typically 0…1; `ProceduralNoise` supplies seeded value-noise fields
//  (fBm optional). "normal" displacement uses the WELDED smooth normal, so a flat-shaded mesh (three vertices a
//  triangle) does not tear along its hard edges — every copy of a point moves identically. For that same reason the
//  field should be a function of position; a field that reads `uv` will crack the mesh at UV seams.
//
//  Guarantees: deterministic; triangle list untouched; normals turned with their faces. Not guaranteed: no remeshing —
//  displacement detail is limited by the mesh's own vertex density (subdivide first); big strengths can self-intersect.
//  CPU only, see Modifier.swift.

import simd

public enum ProceduralNoise {
    /// Seeded 3D value noise in 0…1, trilinear with smoothstep fade. Cheap, band-limited at `frequency` cycles/metre.
    public static func value(_ p: Vec3, seed: UInt64 = 0) -> Double {
        let f = SIMD3<Double>(p.x.rounded(.down), p.y.rounded(.down), p.z.rounded(.down))
        let t = p - f
        let s = t * t * (3 - 2 * t)
        let i = SIMD3<Int64>(Int64(f.x), Int64(f.y), Int64(f.z))
        func h(_ dx: Int64, _ dy: Int64, _ dz: Int64) -> Double {
            func u(_ v: Int64) -> UInt64 { UInt64(bitPattern: v) }
            var x = seed &+ u(i.x &+ dx) &* 0x9E3779B97F4A7C15 &+ u(i.y &+ dy) &* 0xBF58476D1CE4E5B9
                &+ u(i.z &+ dz) &* 0x94D049BB133111EB
            x = (x ^ (x >> 30)) &* 0xBF58476D1CE4E5B9
            x = (x ^ (x >> 27)) &* 0x94D049BB133111EB
            x ^= x >> 31
            return Double(x >> 11) / Double(1 << 53)
        }
        func mix(_ a: Double, _ b: Double, _ k: Double) -> Double { a + (b - a) * k }
        let x00 = mix(h(0, 0, 0), h(1, 0, 0), s.x), x10 = mix(h(0, 1, 0), h(1, 1, 0), s.x)
        let x01 = mix(h(0, 0, 1), h(1, 0, 1), s.x), x11 = mix(h(0, 1, 1), h(1, 1, 1), s.x)
        return mix(mix(x00, x10, s.y), mix(x01, x11, s.y), s.z)
    }

    /// fBm of `value`: `octaves` layers, each `lacunarity`× finer and `gain`× weaker, normalised back to 0…1.
    public static func fbm(_ p: Vec3, frequency: Double, octaves: Int = 4, lacunarity: Double = 2, gain: Double = 0.5,
                           seed: UInt64 = 0) -> Double {
        var sum = 0.0, amp = 1.0, norm = 0.0, f = frequency
        for o in 0..<Swift.max(1, octaves) {
            sum += amp * value(p * f, seed: seed &+ UInt64(o) &* 7919); norm += amp
            amp *= gain; f *= lacunarity
        }
        return sum / norm
    }
}

public struct DisplaceModifier: Modifier {
    public enum Direction: Sendable { case normal, fixed(Vec3) }
    public typealias Field = @Sendable (Vec3, Vec2) -> Double

    public var field: Field
    public var label: String
    public var strength: Double
    public var midlevel: Double
    public var direction: Direction

    public init(label: String = "custom", strength: Double, midlevel: Double = 0.5, direction: Direction = .normal,
                field: @escaping Field) {
        self.label = label; self.strength = strength; self.midlevel = midlevel; self.direction = direction; self.field = field
    }

    /// Seeded fBm noise displacement.
    public static func noise(strength: Double, frequency: Double, octaves: Int = 4, seed: UInt64 = 0,
                             midlevel: Double = 0.5, direction: Direction = .normal) -> DisplaceModifier {
        DisplaceModifier(label: "noise f:\(frequency) o:\(octaves) seed:\(seed)", strength: strength, midlevel: midlevel,
                         direction: direction) { p, _ in ProceduralNoise.fbm(p, frequency: frequency, octaves: octaves, seed: seed) }
    }

    public var summary: String { "Displace(\(label) strength:\(strength) mid:\(midlevel))" }

    public func apply(_ mesh: Mesh3) -> Mesh3 {
        guard !mesh.isEmpty, strength != 0 else { return mesh }
        let w = WeldedMesh(mesh)
        var newP = mesh.positions
        let hasUV = mesh.uvs.count == mesh.positions.count
        for i in newP.indices {
            let id = w.cornerID[i]
            let d: Vec3
            switch direction { case .normal: d = w.normals[id]; case .fixed(let v): d = normalize3(v) }
            // Sample at the welded point so every copy of a vertex gets the same value.
            newP[i] += d * (strength * (field(w.points[id], hasUV ? mesh.uvs[i] : .zero) - midlevel))
        }
        return mesh.moved(to: newP)
    }
}
