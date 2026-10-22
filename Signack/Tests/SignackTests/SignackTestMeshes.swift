import simd
@testable import Signack

/// Small procedural meshes for the GPU-free DAG tests.
enum SignackTestMeshes {
    struct Mesh { var positions: [SIMD3<Float>]; var indices: [UInt32] }

    /// A closed unit icosphere (no seams, no border).
    static func icosphere(subdivisions: Int) -> Mesh {
        let t = Float((1 + 5.0.squareRoot()) / 2)
        var p: [SIMD3<Float>] = [
            SIMD3(-1, t, 0), SIMD3(1, t, 0), SIMD3(-1, -t, 0), SIMD3(1, -t, 0),
            SIMD3(0, -1, t), SIMD3(0, 1, t), SIMD3(0, -1, -t), SIMD3(0, 1, -t),
            SIMD3(t, 0, -1), SIMD3(t, 0, 1), SIMD3(-t, 0, -1), SIMD3(-t, 0, 1),
        ].map { simd_normalize($0) }
        var f: [SIMD3<UInt32>] = [
            SIMD3(0, 11, 5), SIMD3(0, 5, 1), SIMD3(0, 1, 7), SIMD3(0, 7, 10), SIMD3(0, 10, 11),
            SIMD3(1, 5, 9), SIMD3(5, 11, 4), SIMD3(11, 10, 2), SIMD3(10, 7, 6), SIMD3(7, 1, 8),
            SIMD3(3, 9, 4), SIMD3(3, 4, 2), SIMD3(3, 2, 6), SIMD3(3, 6, 8), SIMD3(3, 8, 9),
            SIMD3(4, 9, 5), SIMD3(2, 4, 11), SIMD3(6, 2, 10), SIMD3(8, 6, 7), SIMD3(9, 8, 1),
        ]
        for _ in 0..<subdivisions {
            var mid: [UInt64: UInt32] = [:]
            func m(_ a: UInt32, _ b: UInt32) -> UInt32 {
                let k = a < b ? (UInt64(a) << 32 | UInt64(b)) : (UInt64(b) << 32 | UInt64(a))
                if let i = mid[k] { return i }
                p.append(simd_normalize((p[Int(a)] + p[Int(b)]) * 0.5))
                mid[k] = UInt32(p.count - 1)
                return UInt32(p.count - 1)
            }
            f = f.flatMap { t -> [SIMD3<UInt32>] in
                let a = m(t.x, t.y), b = m(t.y, t.z), c = m(t.z, t.x)
                return [SIMD3(t.x, a, c), SIMD3(t.y, b, a), SIMD3(t.z, c, b), SIMD3(a, b, c)]
            }
        }
        return Mesh(positions: p, indices: f.flatMap { [$0.x, $0.y, $0.z] })
    }

    /// A UV sphere whose u = 0 / u = 1 meridian is duplicated (a texture SEAM: two vertices at one
    /// position), with pole fans — exercises seam locking.
    static func uvSphere(meridians: Int, parallels: Int) -> Mesh {
        var p: [SIMD3<Float>] = []
        for j in 0...parallels {
            let phi = Float(j) / Float(parallels) * .pi
            for i in 0...meridians {
                let th = Float(i % meridians) / Float(meridians) * 2 * .pi   // i == meridians: same spot as 0
                p.append(SIMD3(sin(phi) * cos(th), cos(phi), sin(phi) * sin(th)))
            }
        }
        var idx: [UInt32] = []
        let s = meridians + 1
        for j in 0..<parallels { for i in 0..<meridians {
            let a = UInt32(j * s + i), b = a + 1, c = UInt32((j + 1) * s + i), d = c + 1
            if j != 0 { idx += [a, b, c] }
            if j != parallels - 1 { idx += [b, d, c] }
        } }
        return Mesh(positions: p, indices: idx)
    }

    /// An open, gently waved grid (has a border).
    static func grid(_ n: Int) -> Mesh {
        var p: [SIMD3<Float>] = []
        for j in 0...n { for i in 0...n {
            let x = Float(i) / Float(n), z = Float(j) / Float(n)
            p.append(SIMD3(x, 0.05 * sin(x * 9) * cos(z * 7), z))
        } }
        var idx: [UInt32] = []
        for j in 0..<n { for i in 0..<n {
            let a = UInt32(j * (n + 1) + i), b = a + 1, c = a + UInt32(n + 1), d = c + 1
            idx += [a, c, b, b, c, d]
        } }
        return Mesh(positions: p, indices: idx)
    }

    /// `count` small two-triangle cards scattered in a ball (a crude crown). Atom = card index.
    static func cards(_ count: Int) -> (mesh: Mesh, atoms: [UInt32], cards: [SignackCard]) {
        var p: [SIMD3<Float>] = []
        var idx: [UInt32] = []
        var atoms: [UInt32] = []
        var cards: [SignackCard] = []
        var s: UInt64 = 12345
        func r() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
        for k in 0..<count {
            let c = simd_normalize(SIMD3(r() - 0.5, r() - 0.5, r() - 0.5)) * (0.3 + 0.7 * r()) * 3
            let n = simd_normalize(SIMD3(r() - 0.5, r() - 0.5, r() - 0.5))
            let u = simd_normalize(simd_cross(n, abs(n.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)))
            let v = simd_cross(n, u)
            let h: Float = 0.04 + 0.03 * r()
            let base = UInt32(p.count)
            p += [c - u * h - v * h, c + u * h - v * h, c + u * h + v * h, c - u * h + v * h]
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
            atoms += [UInt32(k), UInt32(k)]
            cards.append(SignackCard(firstTriangle: 2 * k, triangleCount: 2, centroid: c, normal: n,
                                     area: 4 * h * h))
        }
        return (Mesh(positions: p, indices: idx), atoms, cards)
    }
}
