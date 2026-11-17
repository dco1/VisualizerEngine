//  PLY read / write for MeshIO. See MeshIO.swift for what is and isn't guaranteed.
//  Read: ascii, binary_little_endian, binary_big_endian; any scalar type; x y z, nx ny nz, s t / u v /
//  texture_u texture_v, red green blue (uchar → 0…1); faces as a list (n-gons triangulated in-plane, winding
//  kept); other elements are parsed and skipped. Missing normals → crease-smoothed from faces.
//  Write: one mesh (several are merged), binary LE float32 (or ascii doubles), colours if supplied.

import Foundation
import simd

extension MeshIO {

    private enum PType: String {
        case int8, uint8, int16, uint16, int32, uint32, float32, float64
        init?(name: String) {
            switch name {
            case "char", "int8": self = .int8
            case "uchar", "uint8": self = .uint8
            case "short", "int16": self = .int16
            case "ushort", "uint16": self = .uint16
            case "int", "int32": self = .int32
            case "uint", "uint32": self = .uint32
            case "float", "float32": self = .float32
            case "double", "float64": self = .float64
            default: return nil
            }
        }
        var size: Int {
            switch self { case .int8, .uint8: return 1; case .int16, .uint16: return 2
            case .int32, .uint32, .float32: return 4; case .float64: return 8 }
        }
    }
    private struct Prop { var name: String; var type: PType; var listCount: PType? }
    private struct Element { var name: String; var count: Int; var props: [Prop] = [] }

    private struct Cursor {
        enum Mode { case ascii, little, big }
        let b: [UInt8]; var pos = 0; let mode: Mode
        init(_ b: [UInt8], mode: Mode) { self.b = b; self.mode = mode }

        mutating func read(_ t: PType) throws -> Double {
            if mode == .ascii {
                while pos < b.count, b[pos] <= 32 { pos += 1 }
                let s = pos
                while pos < b.count, b[pos] > 32 { pos += 1 }
                guard s < pos, let d = Double(String(decoding: b[s..<pos], as: UTF8.self)), d.isFinite else {
                    throw pos >= b.count ? Failure.truncated("ply body") : Failure.malformed("ply: bad number")
                }
                return d
            }
            guard pos + t.size <= b.count else { throw Failure.truncated("ply body") }
            var u: UInt64 = 0
            for k in 0..<t.size {
                let byte = UInt64(b[pos + (mode == .little ? t.size - 1 - k : k)])
                u = (u << 8) | byte
            }
            pos += t.size
            switch t {
            case .int8: return Double(Int8(truncatingIfNeeded: u))
            case .uint8: return Double(UInt8(truncatingIfNeeded: u))
            case .int16: return Double(Int16(truncatingIfNeeded: u))
            case .uint16: return Double(UInt16(truncatingIfNeeded: u))
            case .int32: return Double(Int32(truncatingIfNeeded: u))
            case .uint32: return Double(UInt32(truncatingIfNeeded: u))
            case .float32: return Double(Float(bitPattern: UInt32(truncatingIfNeeded: u)))
            case .float64: return Double(bitPattern: u)
            }
        }
    }

    static func readPLY(_ data: Data) throws -> Result {
        let bytes = [UInt8](data)
        guard bytes.starts(with: Array("ply".utf8)) else { throw Failure.malformed("ply: missing magic") }
        let marker = Array("end_header".utf8)
        var h = -1
        search: for i in 0..<max(0, bytes.count - marker.count) where bytes[i] == marker[0] {
            if Array(bytes[i..<(i + marker.count)]) == marker { h = i; break search }
        }
        guard h >= 0 else { throw Failure.truncated("ply header") }
        var bodyStart = h + marker.count
        while bodyStart < bytes.count, bytes[bodyStart] != 10 { bodyStart += 1 }
        bodyStart += 1                                         // past the newline (CRLF: the \r was skipped above)
        let header = String(decoding: bytes[0..<h], as: UTF8.self)

        var mode: Cursor.Mode? = nil
        var elements: [Element] = []
        for line in header.split(whereSeparator: \.isNewline) {
            let t = line.split(separator: " ")
            guard let k = t.first else { continue }
            switch k {
            case "format":
                guard t.count >= 2 else { throw Failure.malformed("ply: format") }
                switch t[1] {
                case "ascii": mode = .ascii
                case "binary_little_endian": mode = .little
                case "binary_big_endian": mode = .big
                default: throw Failure.unsupported("ply format \(t[1])")
                }
            case "element":
                guard t.count == 3, let n = Int(t[2]), n >= 0 else { throw Failure.malformed("ply: element line") }
                elements.append(Element(name: String(t[1]), count: n))
            case "property":
                guard !elements.isEmpty else { throw Failure.malformed("ply: property before element") }
                if t.count == 5, t[1] == "list" {
                    guard let c = PType(name: String(t[2])), let it = PType(name: String(t[3])) else { throw Failure.unsupported("ply list type") }
                    elements[elements.count - 1].props.append(Prop(name: String(t[4]), type: it, listCount: c))
                } else if t.count == 3, let ty = PType(name: String(t[1])) {
                    elements[elements.count - 1].props.append(Prop(name: String(t[2]), type: ty, listCount: nil))
                } else { throw Failure.malformed("ply: property line '\(line)'") }
            default: break
            }
        }
        guard let mode else { throw Failure.malformed("ply: no format line") }
        guard bodyStart <= bytes.count else { throw Failure.truncated("ply body") }
        var cur = Cursor(Array(bytes[bodyStart...]), mode: mode)

        var mesh = Mesh3(); var colors: [SIMD3<Float>] = []
        var hasN = false, hasC = false
        var faces: [[UInt32]] = []
        for el in elements {
            guard el.count < 100_000_000 else { throw Failure.malformed("ply: absurd element count") }
            if el.name == "vertex" {
                let ix = Dictionary(el.props.enumerated().map { ($1.name, $0) }, uniquingKeysWith: { a, _ in a })
                func col(_ names: [String]) -> Int? { names.compactMap { ix[$0] }.first }
                let (cx, cy, cz) = (ix["x"], ix["y"], ix["z"])
                guard let cx, let cy, let cz else { throw Failure.malformed("ply: vertex without x y z") }
                let cn = (ix["nx"], ix["ny"], ix["nz"])
                let cu = col(["s", "u", "texture_u", "texture_s"]), cv = col(["t", "v", "texture_v", "texture_t"])
                let cr = (ix["red"], ix["green"], ix["blue"])
                hasN = cn.0 != nil && cn.1 != nil && cn.2 != nil
                hasC = cr.0 != nil && cr.1 != nil && cr.2 != nil
                mesh.positions.reserveCapacity(el.count)
                var row = [Double](repeating: 0, count: el.props.count)
                for _ in 0..<el.count {
                    for (k, p) in el.props.enumerated() {
                        if let lc = p.listCount { let n = Int(try cur.read(lc)); for _ in 0..<max(0, n) { _ = try cur.read(p.type) } }
                        else { row[k] = try cur.read(p.type) }
                    }
                    mesh.positions.append(Vec3(row[cx], row[cy], row[cz]))
                    mesh.uvs.append(Vec2(cu.map { row[$0] } ?? 0, cv.map { row[$0] } ?? 0))
                    if hasN {
                        let n = Vec3(row[cn.0!], row[cn.1!], row[cn.2!])
                        mesh.normals.append(len3(n) > 1e-300 ? normalize3(n) : Vec3(0, 0, 1))
                    } else { mesh.normals.append(Vec3(0, 0, 1)) }
                    if hasC {
                        func c(_ i: Int) -> Float {
                            let v = Float(row[i]); return el.props[i].type.size == 1 ? v / 255 : v
                        }
                        colors.append(SIMD3<Float>(c(cr.0!), c(cr.1!), c(cr.2!)))
                    }
                }
            } else if el.name == "face" {
                for _ in 0..<el.count {
                    for p in el.props {
                        if let lc = p.listCount {
                            let n = Int(try cur.read(lc))
                            guard n >= 0, n < 1_000_000 else { throw Failure.malformed("ply: face size") }
                            var f: [UInt32] = []; f.reserveCapacity(n)
                            for _ in 0..<n {
                                let v = try cur.read(p.type)
                                guard v >= 0, v < Double(mesh.positions.count) else { throw Failure.malformed("ply: face index \(v) out of range") }
                                f.append(UInt32(v))
                            }
                            if p.name == "vertex_indices" || p.name == "vertex_index" { if f.count >= 3 { faces.append(f) } }
                        } else { _ = try cur.read(p.type) }
                    }
                }
            } else {                                           // unknown element: parse and drop
                for _ in 0..<el.count {
                    for p in el.props {
                        if let lc = p.listCount { let n = Int(try cur.read(lc)); for _ in 0..<max(0, n) { _ = try cur.read(p.type) } }
                        else { _ = try cur.read(p.type) }
                    }
                }
            }
        }
        for f in faces { appendPolygon(f, positions: mesh.positions, to: &mesh.indices) }
        if !hasN { mesh = smoothedFromFaces(mesh, crease: 40) }
        var r = Result()
        r.meshes = [mesh]; r.names = [""]; r.materialNames = [nil]; r.colors = [hasC ? colors : []]
        return r
    }

    static func writePLY(_ meshes: [Mesh3], options: WriteOptions) -> Data {
        var m = Mesh3(); var cols: [SIMD3<Float>] = []; var haveC = !meshes.isEmpty
        for (i, x) in meshes.enumerated() {
            var mm = x
            if mm.normals.count != mm.positions.count { mm.normals = faceNormalsPerVertex(mm) }
            if mm.uvs.count != mm.positions.count { mm.uvs = [Vec2](repeating: .zero, count: mm.positions.count) }
            m.append(mm)
            if i < options.colors.count, options.colors[i].count == x.positions.count { cols += options.colors[i] } else { haveC = false }
        }
        let binary = options.binary
        var head = "ply\nformat \(binary ? "binary_little_endian" : "ascii") 1.0\ncomment VisualizerMaterials MeshIO\n"
        let ft = binary ? "float" : "double"
        head += "element vertex \(m.positions.count)\n"
        for p in ["x", "y", "z", "nx", "ny", "nz", "s", "t"] { head += "property \(ft) \(p)\n" }
        if haveC { for p in ["red", "green", "blue"] { head += "property uchar \(p)\n" } }
        head += "element face \(m.indices.count / 3)\nproperty list uchar int vertex_indices\nend_header\n"
        func byte(_ f: Float) -> UInt8 { UInt8(max(0, min(255, (f * 255).rounded()))) }
        if !binary {
            var s = head
            for i in 0..<m.positions.count {
                let p = m.positions[i], n = m.normals[i], t = m.uvs[i]
                s += "\(p.x) \(p.y) \(p.z) \(n.x) \(n.y) \(n.z) \(t.x) \(t.y)"
                if haveC { s += " \(byte(cols[i].x)) \(byte(cols[i].y)) \(byte(cols[i].z))" }
                s += "\n"
            }
            var t = 0
            while t + 2 < m.indices.count { s += "3 \(m.indices[t]) \(m.indices[t + 1]) \(m.indices[t + 2])\n"; t += 3 }
            return utf8(s)
        }
        var d = utf8(head)
        func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func putF(_ f: Double) { put(Float(f).bitPattern) }
        for i in 0..<m.positions.count {
            let p = m.positions[i], n = m.normals[i], t = m.uvs[i]
            for v in [p.x, p.y, p.z, n.x, n.y, n.z, t.x, t.y] { putF(v) }
            if haveC { d.append(byte(cols[i].x)); d.append(byte(cols[i].y)); d.append(byte(cols[i].z)) }
        }
        var t = 0
        while t + 2 < m.indices.count {
            d.append(3); put(Int32(m.indices[t])); put(Int32(m.indices[t + 1])); put(Int32(m.indices[t + 2])); t += 3
        }
        return d
    }
}
