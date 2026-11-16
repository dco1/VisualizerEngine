//  STL read / write for MeshIO. See MeshIO.swift for what is and isn't guaranteed.
//  Read: binary is recognised by SIZE (84 + 50·count == length), so a binary file whose 80-byte header begins
//  with "solid" is read as binary; otherwise a "solid" text file is read as ascii (several solids → several
//  meshes). Facets keep their winding; the stored facet normal is kept unless it is zero or disagrees in sign
//  with the winding's normal (then the winding wins — the winding is what the engine culls by). Vertices are
//  unshared (3 a triangle) unless `ReadOptions.weldAndSmooth`. Write: one mesh (several merged for binary,
//  one solid each for ascii); facet normals from the winding.

import Foundation
import simd

extension MeshIO {

    static func isBinarySTL(_ data: Data) -> Bool {
        guard data.count >= 84 else { return false }
        let n = data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 80, as: UInt32.self)) }
        return 84 + 50 * Int(n) == data.count
    }

    static func readSTL(_ data: Data, options: ReadOptions) throws -> Result {
        var r = Result()
        func finish(_ m: Mesh3, name: String) {
            var m = m
            if options.weldAndSmooth { m = weldedSmooth(m, crease: options.creaseDegrees) }
            r.meshes.append(m); r.names.append(name); r.materialNames.append(nil); r.colors.append([])
        }
        if isBinarySTL(data) {
            let count = data.withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 80, as: UInt32.self))) }
            var m = Mesh3()
            m.positions.reserveCapacity(count * 3)
            data.withUnsafeBytes { raw in
                func f(_ o: Int) -> Double { Double(Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: o, as: UInt32.self)))) }
                for i in 0..<count {
                    let o = 84 + i * 50
                    let n = Vec3(f(o), f(o + 4), f(o + 8))
                    appendFacet(&m, n, Vec3(f(o + 12), f(o + 16), f(o + 20)), Vec3(f(o + 24), f(o + 28), f(o + 32)), Vec3(f(o + 36), f(o + 40), f(o + 44)))
                }
            }
            guard count > 0 else { throw Failure.malformed("stl: no facets") }
            finish(m, name: "")
            return r
        }
        guard data.starts(with: Array("solid".utf8)) else { throw Failure.malformed("stl: neither binary (size) nor ascii (solid)") }
        let text = String(decoding: data, as: UTF8.self)
        let tok = text.split(whereSeparator: { $0 == " " || $0 == "\t" || $0.isNewline })
        var i = 0
        var m = Mesh3(); var name = ""; var open = false
        func num() throws -> Double {
            guard i < tok.count, let d = Double(tok[i]), d.isFinite else { throw Failure.malformed("stl: bad number near token \(i)") }
            i += 1; return d
        }
        func vec() throws -> Vec3 { Vec3(try num(), try num(), try num()) }
        while i < tok.count {
            switch tok[i] {
            case "solid":
                i += 1
                var parts: [Substring] = []
                while i < tok.count, tok[i] != "facet", tok[i] != "endsolid" { parts.append(tok[i]); i += 1 }
                name = parts.joined(separator: " "); open = true
            case "facet":
                i += 1
                guard i < tok.count, tok[i] == "normal" else { throw Failure.malformed("stl: expected 'normal'") }
                i += 1; let n = try vec()
                guard i + 1 < tok.count, tok[i] == "outer", tok[i + 1] == "loop" else { throw Failure.malformed("stl: expected 'outer loop'") }
                i += 2
                var v: [Vec3] = []
                for _ in 0..<3 {
                    guard i < tok.count, tok[i] == "vertex" else { throw Failure.malformed("stl: expected 'vertex'") }
                    i += 1; v.append(try vec())
                }
                guard i < tok.count, tok[i] == "endloop" else { throw Failure.malformed("stl: expected 'endloop'") }
                i += 1
                guard i < tok.count, tok[i] == "endfacet" else { throw Failure.malformed("stl: expected 'endfacet'") }
                i += 1
                appendFacet(&m, n, v[0], v[1], v[2])
            case "endsolid":
                i += 1
                while i < tok.count, tok[i] != "solid" { i += 1 }   // skip the name
                if !m.indices.isEmpty { finish(m, name: name) }
                m = Mesh3(); open = false
            default: throw Failure.malformed("stl: unexpected token '\(tok[i])'")
            }
        }
        if !m.indices.isEmpty { finish(m, name: name) }          // missing endsolid: keep what parsed
        _ = open
        guard !r.meshes.isEmpty else { throw Failure.malformed("stl: no facets") }
        return r
    }

    private static func appendFacet(_ m: inout Mesh3, _ stored: Vec3, _ a: Vec3, _ b: Vec3, _ c: Vec3) {
        let raw = cross3(b - a, c - a)
        var n = len3(raw) > 1e-300 ? normalize3(raw) : Vec3(0, 0, 1)
        if len3(stored) > 1e-9, len3(raw) > 1e-300, dot3(normalize3(stored), n) > 0 { n = normalize3(stored) }
        let base = UInt32(m.positions.count)
        m.positions += [a, b, c]; m.normals += [n, n, n]; m.uvs += [.zero, .zero, .zero]
        m.indices += [base, base + 1, base + 2]
    }

    /// Weld positions that agree to 1e-6 relative to the bounds, then crease-smooth.
    static func weldedSmooth(_ m: Mesh3, crease: Double) -> Mesh3 {
        guard let b = m.bounds else { return m }
        let q = max(1e-9, len3(b.max - b.min) * 1e-6)
        struct K: Hashable { let x, y, z: Int64 }
        var map: [K: UInt32] = [:]; var out = Mesh3()
        var remap = [UInt32](repeating: 0, count: m.positions.count)
        for (i, p) in m.positions.enumerated() {
            let k = K(x: Int64((p.x / q).rounded()), y: Int64((p.y / q).rounded()), z: Int64((p.z / q).rounded()))
            if let j = map[k] { remap[i] = j } else {
                let j = UInt32(out.positions.count); map[k] = j; remap[i] = j
                out.positions.append(p); out.uvs.append(.zero); out.normals.append(.zero)
            }
        }
        var t = 0
        while t + 2 < m.indices.count {
            let a = remap[Int(m.indices[t])], bb = remap[Int(m.indices[t + 1])], c = remap[Int(m.indices[t + 2])]
            if a != bb, bb != c, a != c { out.indices += [a, bb, c] }     // welding can collapse a sliver
            t += 3
        }
        // smoothed() averages by position, so welded or not the average is the same; the weld just shares vertices.
        var withN = out; withN.normals = faceNormalsPerVertex(out)
        return withN.smoothed(creaseDegrees: crease)
    }

    static func writeSTL(_ meshes: [Mesh3], options: WriteOptions) -> Data {
        func facets(_ m: Mesh3, _ body: (Vec3, Vec3, Vec3, Vec3) -> Void) {
            var t = 0
            while t + 2 < m.indices.count {
                let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])], c = m.positions[Int(m.indices[t + 2])]
                let raw = cross3(b - a, c - a)
                body(len3(raw) > 1e-300 ? normalize3(raw) : Vec3(0, 0, 0), a, b, c); t += 3
            }
        }
        if !options.binary {
            var s = ""
            for (i, m) in meshes.enumerated() {
                let name = i < options.names.count && !options.names[i].isEmpty ? options.names[i] : "mesh\(i)"
                s += "solid \(name)\n"
                facets(m) { n, a, b, c in
                    s += "  facet normal \(n.x) \(n.y) \(n.z)\n    outer loop\n"
                    for v in [a, b, c] { s += "      vertex \(v.x) \(v.y) \(v.z)\n" }
                    s += "    endloop\n  endfacet\n"
                }
                s += "endsolid \(name)\n"
            }
            return utf8(s)
        }
        var all = Mesh3(); for m in meshes { all.append(m) }
        var d = Data(count: 80)
        let tag = Array("VisualizerMaterials MeshIO binary STL".utf8)
        d.replaceSubrange(0..<tag.count, with: tag)             // deliberately not "solid…"
        func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func putF(_ f: Double) { put(Float(f).bitPattern) }
        put(UInt32(all.indices.count / 3))
        facets(all) { n, a, b, c in
            for v in [n, a, b, c] { putF(v.x); putF(v.y); putF(v.z) }
            put(UInt16(0))
        }
        return d
    }
}
