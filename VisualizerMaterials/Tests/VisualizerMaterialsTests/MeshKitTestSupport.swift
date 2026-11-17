import XCTest
@testable import VisualizerMaterials

/// Shared fixtures and audits for the Mesh3 remesh / decimate / weld tests.
enum MeshKit {
    /// Indexed icosphere (radius r, smooth normals), 20·4^levels triangles.
    static func icosphere(radius r: Double = 1, levels: Int) -> Mesh3 {
        let t = (1 + 5.0.squareRoot()) / 2
        var p: [Vec3] = [Vec3(-1, t, 0), Vec3(1, t, 0), Vec3(-1, -t, 0), Vec3(1, -t, 0), Vec3(0, -1, t), Vec3(0, 1, t),
                         Vec3(0, -1, -t), Vec3(0, 1, -t), Vec3(t, 0, -1), Vec3(t, 0, 1), Vec3(-t, 0, -1), Vec3(-t, 0, 1)].map(normalize3)
        var f: [[Int]] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                          [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
        for _ in 0 ..< levels {
            var cache: [Int64: Int] = [:]
            func mid(_ a: Int, _ b: Int) -> Int {
                let k = Int64(min(a, b)) << 32 | Int64(max(a, b))
                if let m = cache[k] { return m }
                p.append(normalize3(p[a] + p[b])); cache[k] = p.count - 1; return p.count - 1
            }
            var nf: [[Int]] = []
            for q in f {
                let a = mid(q[0], q[1]), b = mid(q[1], q[2]), c = mid(q[2], q[0])
                nf += [[q[0], a, c], [q[1], b, a], [q[2], c, b], [a, b, c]]
            }
            f = nf
        }
        var m = Mesh3()
        m.positions = p.map { $0 * r }; m.normals = p
        m.uvs = [Vec2](repeating: .zero, count: p.count)
        m.indices = f.flatMap { $0.map { UInt32($0) } }
        return m
    }

    /// Flat grid in XZ, n×n quads, facing +Y, UV = (x, z) in 0…1.
    static func plane(n: Int, size: Double = 1) -> Mesh3 {
        var m = Mesh3()
        for j in 0 ... n { for i in 0 ... n {
            let u = Double(i) / Double(n), v = Double(j) / Double(n)
            m.positions.append(Vec3(u * size, 0, v * size)); m.normals.append(Vec3(0, 1, 0)); m.uvs.append(Vec2(u, v))
        } }
        for j in 0 ..< n { for i in 0 ..< n {
            let a = UInt32(j * (n + 1) + i), b = a + 1, c = a + UInt32(n + 1), d = c + 1
            m.indices += [a, c, b, b, c, d]
        } }
        return m
    }

    /// Unit box, each face an n×n grid of quads (flat soup: every triangle owns its vertices).
    static func box(n: Int, size: Double = 1) -> Mesh3 {
        var m = Mesh3()
        let faces: [(Vec3, Vec3, Vec3)] = [(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)), (Vec3(1, 0, 1), Vec3(-1, 0, 0), Vec3(0, 1, 0)),
                                           (Vec3(0, 0, 1), Vec3(0, 0, -1), Vec3(0, 1, 0)), (Vec3(1, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0)),
                                           (Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)), (Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0, 0, -1))]
        for (o, u, v) in faces {
            var out = normalize3(cross3(u, v))
            if dot3(out, o + (u + v) * 0.5 - Vec3(0.5, 0.5, 0.5)) < 0 { out = -out }
            for j in 0 ..< n { for i in 0 ..< n {
                func pt(_ a: Int, _ b: Int) -> Vec3 { (o + u * (Double(a) / Double(n)) + v * (Double(b) / Double(n))) * size }
                m.addQuad(pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1), outward: out)
            } }
        }
        return m
    }

    static func volume(_ m: Mesh3) -> Double {
        var v = 0.0, t = 0
        while t + 2 < m.indices.count {
            v += dot3(m.positions[Int(m.indices[t])], cross3(m.positions[Int(m.indices[t + 1])], m.positions[Int(m.indices[t + 2])])) / 6
            t += 3
        }
        return v
    }

    /// Edges (by vertex index) not shared by exactly two triangles, with opposite direction twins.
    static func openEdges(_ m: Mesh3) -> Int {
        var count: [Int64: Int] = [:], dir: [Int64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for k in 0 ..< 3 {
                let a = Int(m.indices[t + k]), b = Int(m.indices[t + (k + 1) % 3])
                count[Int64(min(a, b)) << 32 | Int64(max(a, b)), default: 0] += 1
                dir[Int64(a) << 32 | Int64(b), default: 0] += 1
            }
            t += 3
        }
        let badCount = count.values.filter { $0 != 2 }.count
        let badDir = dir.values.filter { $0 != 1 }.count
        return badCount + badDir
    }

    /// Boundary half-edges (edge with no twin), by index.
    static func boundaryVertices(_ m: Mesh3) -> Set<Int> {
        var dir: [Int64: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            for k in 0 ..< 3 { dir[Int64(m.indices[t + k]) << 32 | Int64(m.indices[t + (k + 1) % 3]), default: 0] += 1 }
            t += 3
        }
        var s = Set<Int>()
        for key in dir.keys where dir[(key & 0xFFFF_FFFF) << 32 | (key >> 32)] == nil { s.insert(Int(key >> 32)); s.insert(Int(key & 0xFFFF_FFFF)) }
        return s
    }

    static func area(_ m: Mesh3) -> Double {
        var a = 0.0, t = 0
        while t + 2 < m.indices.count {
            let p = m.positions[Int(m.indices[t])]
            a += 0.5 * len3(cross3(m.positions[Int(m.indices[t + 1])] - p, m.positions[Int(m.indices[t + 2])] - p)); t += 3
        }
        return a
    }

    static func edgeLengths(_ m: Mesh3) -> [Double] {
        var seen = Set<Int64>(), out: [Double] = []
        var t = 0
        while t + 2 < m.indices.count {
            for k in 0 ..< 3 {
                let a = Int(m.indices[t + k]), b = Int(m.indices[t + (k + 1) % 3])
                if seen.insert(Int64(min(a, b)) << 32 | Int64(max(a, b))).inserted { out.append(dist3(m.positions[a], m.positions[b])) }
            }
            t += 3
        }
        return out
    }
}
