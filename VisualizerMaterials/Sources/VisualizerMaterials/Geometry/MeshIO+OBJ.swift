//  Wavefront OBJ read / write for MeshIO. See MeshIO.swift for what is and isn't guaranteed.
//  Read: v (optionally + r g b), vt, vn, f in every index form (v, v/vt, v//vn, v/vt/vn, negative = relative),
//  n-gons triangulated in-plane (winding kept), o / g / usemtl start a new Mesh3, mtllib names kept.
//  A file without vn gets face-derived, crease-smoothed normals. Write: v/vt/vn/f with usemtl + optional MTL.

import Foundation
import simd

extension MeshIO {

    static func readOBJ(_ data: Data) throws -> Result {
        guard !data.isEmpty else { throw Failure.malformed("empty file") }
        let text = String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self)
        var P: [Vec3] = [], C: [SIMD3<Float>] = [], T: [Vec2] = [], N: [Vec3] = []
        var anyColor = false

        struct Key: Hashable { let v: Int, t: Int, n: Int }
        struct Builder {
            var name = "", material: String? = nil
            var mesh = Mesh3(), colors: [SIMD3<Float>] = []
            var map: [Key: UInt32] = [:]
            var sawNormals = false
            var missing: [Int] = []
        }
        var cur = Builder()
        var result = Result()
        var pendingName = ""
        var curMaterial: String? = nil

        func flush() {
            guard !cur.mesh.indices.isEmpty else { cur.mesh = Mesh3(); cur.map = [:]; cur.colors = []; return }
            var m = cur.mesh
            if !cur.sawNormals { m = smoothedFromFaces(m, crease: 40) }
            else if !cur.missing.isEmpty {                       // mixed: corners with no vn get their face's
                let fn = faceNormalsPerVertex(m)
                for i in cur.missing { m.normals[i] = fn[i] }
            }
            result.meshes.append(m); result.names.append(cur.name); result.materialNames.append(cur.material)
            result.colors.append(anyColor ? cur.colors : [])
            cur.mesh = Mesh3(); cur.map = [:]; cur.colors = []; cur.sawNormals = false; cur.missing = []
        }
        func startMesh(name: String?) {
            flush()
            if let name { pendingName = name }
            cur.name = pendingName; cur.material = curMaterial
        }

        func num(_ s: Substring, _ line: Int) throws -> Double {
            guard let d = Double(s), d.isFinite else { throw Failure.malformed("line \(line): bad number '\(s)'") }
            return d
        }
        func resolve(_ s: Substring, count: Int, line: Int) throws -> Int {
            guard let i = Int(s), i != 0 else { throw Failure.malformed("line \(line): bad index '\(s)'") }
            let r = i > 0 ? i - 1 : count + i
            guard r >= 0, r < count else { throw Failure.malformed("line \(line): index \(i) out of range (\(count))") }
            return r
        }

        var lineNo = 0
        var pending = ""
        for raw in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            lineNo += 1
            var line = pending + raw; pending = ""
            if line.hasSuffix("\\") { pending = String(line.dropLast()) + " "; continue }
            if let h = line.firstIndex(of: "#") { line = String(line[..<h]) }
            let tok = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            guard let cmd = tok.first else { continue }
            switch cmd {
            case "v":
                guard tok.count >= 4 else { throw Failure.malformed("line \(lineNo): vertex needs 3 numbers") }
                P.append(Vec3(try num(tok[1], lineNo), try num(tok[2], lineNo), try num(tok[3], lineNo)))
                if tok.count >= 7 {
                    anyColor = true
                    C.append(SIMD3<Float>(Float(try num(tok[4], lineNo)), Float(try num(tok[5], lineNo)), Float(try num(tok[6], lineNo))))
                } else { C.append(SIMD3<Float>(1, 1, 1)) }
            case "vt":
                guard tok.count >= 2 else { throw Failure.malformed("line \(lineNo): vt needs a number") }
                T.append(Vec2(try num(tok[1], lineNo), tok.count > 2 ? try num(tok[2], lineNo) : 0))
            case "vn":
                guard tok.count >= 4 else { throw Failure.malformed("line \(lineNo): vn needs 3 numbers") }
                let n = Vec3(try num(tok[1], lineNo), try num(tok[2], lineNo), try num(tok[3], lineNo))
                N.append(len3(n) > 1e-300 ? normalize3(n) : Vec3(0, 0, 1))
            case "o", "g":
                let name = tok.dropFirst().joined(separator: " ")
                // "g" with the same name as the open mesh (or an empty group) doesn't split a mesh that has no faces yet.
                startMesh(name: name)
            case "usemtl":
                let name = tok.dropFirst().joined(separator: " ")
                curMaterial = (name.isEmpty || name == "off" || name == "(null)") ? nil : name
                if cur.material != curMaterial {
                    if cur.mesh.indices.isEmpty { cur.material = curMaterial } else { startMesh(name: nil) }
                }
            case "mtllib":
                result.materialLibraries.append(contentsOf: tok.dropFirst().map(String.init))
            case "f":
                guard tok.count >= 4 else { throw Failure.malformed("line \(lineNo): face needs 3+ vertices") }
                var local: [UInt32] = []
                for corner in tok.dropFirst() {
                    let parts = corner.split(separator: "/", omittingEmptySubsequences: false)
                    guard parts.count >= 1, parts.count <= 3 else { throw Failure.malformed("line \(lineNo): bad face corner '\(corner)'") }
                    let v = try resolve(parts[0], count: P.count, line: lineNo)
                    let t = parts.count > 1 && !parts[1].isEmpty ? try resolve(parts[1], count: T.count, line: lineNo) : -1
                    let n = parts.count > 2 && !parts[2].isEmpty ? try resolve(parts[2], count: N.count, line: lineNo) : -1
                    let key = Key(v: v, t: t, n: n)
                    if let i = cur.map[key] { local.append(i); continue }
                    let i = UInt32(cur.mesh.positions.count)
                    cur.mesh.positions.append(P[v])
                    cur.mesh.uvs.append(t >= 0 ? T[t] : .zero)
                    cur.mesh.normals.append(n >= 0 ? N[n] : Vec3(0, 0, 1))
                    cur.colors.append(C[v])
                    if n >= 0 { cur.sawNormals = true } else { cur.missing.append(Int(i)) }
                    cur.map[key] = i; local.append(i)
                }
                appendPolygon(local, positions: cur.mesh.positions, to: &cur.mesh.indices)
            default: break   // s, l, p, cstype, … ignored
            }
        }
        flush()
        guard !result.meshes.isEmpty || !P.isEmpty else { throw Failure.malformed("no geometry") }
        return result
    }

    /// OBJ text, and the MTL it names (one `newmtl` per distinct material name, plain grey).
    public static func writeOBJ(_ meshes: [Mesh3], options: WriteOptions = .init()) -> (obj: Data, mtl: Data) {
        var s = "# Written by VisualizerMaterials MeshIO\n"
        if let m = options.mtlFileName { s += "mtllib \(m)\n" }
        var vBase = 1
        var materials: [String] = []
        for (mi, mesh) in meshes.enumerated() {
            let name = mi < options.names.count && !options.names[mi].isEmpty ? options.names[mi] : "mesh\(mi)"
            s += "o \(name)\n"
            if let mat = mi < options.materialNames.count ? options.materialNames[mi] : nil {
                s += "usemtl \(mat)\n"
                if !materials.contains(mat) { materials.append(mat) }
            } else if materials.count > 0 { s += "usemtl (null)\n" }     // usemtl persists; reset it
            let hasUV = hasParallelUVs(mesh), hasN = hasParallelNormals(mesh)
            for p in mesh.positions { s += "v \(p.x) \(p.y) \(p.z)\n" }
            if hasUV { for t in mesh.uvs { s += "vt \(t.x) \(t.y)\n" } }
            if hasN { for n in mesh.normals { s += "vn \(n.x) \(n.y) \(n.z)\n" } }
            var t = 0
            while t + 2 < mesh.indices.count {
                s += "f"
                for k in 0..<3 {
                    let i = Int(mesh.indices[t + k]) + vBase
                    switch (hasUV, hasN) {
                    case (true, true): s += " \(i)/\(i)/\(i)"
                    case (true, false): s += " \(i)/\(i)"
                    case (false, true): s += " \(i)//\(i)"
                    default: s += " \(i)"
                    }
                }
                s += "\n"; t += 3
            }
            vBase += mesh.positions.count
        }
        var mtl = "# MTL names written by VisualizerMaterials MeshIO\n"
        for m in materials { mtl += "newmtl \(m)\nKd 0.8 0.8 0.8\nKs 0 0 0\nNs 10\nd 1\nillum 1\n\n" }
        return (utf8(s), utf8(mtl))
    }
}
