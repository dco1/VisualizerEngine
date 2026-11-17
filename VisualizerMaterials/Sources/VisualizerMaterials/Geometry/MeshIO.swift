//  MeshIO — read / write Mesh3 as OBJ (+MTL names), PLY (ascii, binary LE/BE read, binary LE write) and STL
//  (ascii + binary). Data in, Data out; the URL wrappers are thin.
//
//  Guarantees: triangle winding is preserved exactly (indices are never reordered; n-gons are triangulated in
//  their own plane so the triangles keep the polygon's winding). Mesh3's parallel arrays always line up
//  (positions == normals == uvs). Normals present in the file are kept (normalised); absent ones are made
//  from the faces, crease-smoothed at `creaseDegrees` (Mesh3.smoothed). Malformed input throws `MeshIO.Failure`,
//  it never traps (indices are range-checked, binary reads are bounds-checked).
//  Doesn't: PLY/STL have no objects or groups, so writing several meshes to them merges the meshes. STL and
//  binary PLY store Float32, so positions round-trip to ~1e-7 relative. Vertex colours are carried beside the
//  meshes (Mesh3 has no colour channel). OBJ: no curves/surfaces/smoothing groups; MTL is names only.
//  CPU: file parsing is serial and branchy, not per-vertex number crunching, so there is no GPU twin.

import Foundation
import simd

public enum MeshIO {

    public enum Format: String, Sendable, CaseIterable {
        case obj, ply, stl
        public init?(pathExtension: String) { self.init(rawValue: pathExtension.lowercased()) }
    }

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case unknownFormat
        case malformed(String)
        case truncated(String)
        case unsupported(String)
        public var description: String {
            switch self {
            case .unknownFormat: return "MeshIO: format not recognised"
            case .malformed(let s): return "MeshIO: malformed — \(s)"
            case .truncated(let s): return "MeshIO: truncated — \(s)"
            case .unsupported(let s): return "MeshIO: unsupported — \(s)"
            }
        }
    }

    /// What a read returns. All arrays are parallel to `meshes`.
    public struct Result: Sendable {
        public var meshes: [Mesh3] = []
        /// OBJ `o`/`g` name (empty if none); STL `solid` name; PLY empty.
        public var names: [String] = []
        /// OBJ `usemtl` name in force for the mesh, else nil.
        public var materialNames: [String?] = []
        /// OBJ `mtllib` file names, in order.
        public var materialLibraries: [String] = []
        /// Per-vertex RGB 0…1 when the file carries colours (PLY red/green/blue, OBJ `v x y z r g b`), else `[]`.
        public var colors: [[SIMD3<Float>]] = []
        public init() {}
        /// All meshes appended into one.
        public var merged: Mesh3 { var m = Mesh3(); for x in meshes { m.append(x) }; return m }
    }

    public struct ReadOptions: Sendable {
        /// STL only: weld coincident vertices and smooth the normals (otherwise flat facet normals).
        public var weldAndSmooth = false
        public var creaseDegrees = 40.0
        public init(weldAndSmooth: Bool = false, creaseDegrees: Double = 40) {
            self.weldAndSmooth = weldAndSmooth; self.creaseDegrees = creaseDegrees
        }
    }

    public struct WriteOptions: Sendable {
        public var names: [String] = []
        public var materialNames: [String?] = []
        /// Per mesh, per vertex (PLY only, merged in mesh order). Ignored if a mesh's count doesn't match.
        public var colors: [[SIMD3<Float>]] = []
        /// PLY / STL: binary (little endian) when true.
        public var binary = true
        /// OBJ: emit `mtllib <name>` (the MTL itself comes from `writeOBJ`).
        public var mtlFileName: String? = nil
        public init() {}
    }

    // MARK: API

    public static func read(data: Data, format: Format? = nil, options: ReadOptions = .init()) throws -> Result {
        let f = format ?? detect(data)
        guard let f else { throw Failure.unknownFormat }
        switch f {
        case .obj: return try readOBJ(data)
        case .ply: return try readPLY(data)
        case .stl: return try readSTL(data, options: options)
        }
    }

    public static func read(url: URL, options: ReadOptions = .init()) throws -> Result {
        let data = try Data(contentsOf: url)
        return try read(data: data, format: Format(pathExtension: url.pathExtension), options: options)
    }

    public static func write(_ meshes: [Mesh3], format: Format, options: WriteOptions = .init()) -> Data {
        switch format {
        case .obj: return writeOBJ(meshes, options: options).obj
        case .ply: return writePLY(meshes, options: options)
        case .stl: return writeSTL(meshes, options: options)
        }
    }

    public static func write(_ mesh: Mesh3, format: Format, options: WriteOptions = .init()) -> Data {
        write([mesh], format: format, options: options)
    }

    public static func write(_ meshes: [Mesh3], to url: URL, format: Format? = nil, options: WriteOptions = .init()) throws {
        guard let f = format ?? Format(pathExtension: url.pathExtension) else { throw Failure.unknownFormat }
        try write(meshes, format: f, options: options).write(to: url)
    }

    /// Sniff: PLY by magic, STL by size rule or "solid", OBJ if it looks like text with v/f lines.
    public static func detect(_ data: Data) -> Format? {
        if data.starts(with: Array("ply".utf8)) { return .ply }
        if isBinarySTL(data) || data.starts(with: Array("solid".utf8)) { return .stl }
        let head = String(decoding: data.prefix(4096), as: UTF8.self)
        for line in head.split(whereSeparator: \.isNewline) {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("v ") || t.hasPrefix("f ") || t.hasPrefix("o ") || t.hasPrefix("mtllib") { return .obj }
        }
        return nil
    }

    // MARK: shared helpers

    /// Fill `normals` from the faces and smooth them (used when a file has none).
    static func smoothedFromFaces(_ m: Mesh3, crease: Double) -> Mesh3 {
        var m = m
        m.normals = faceNormalsPerVertex(m)
        return m.smoothed(creaseDegrees: crease)
    }

    static func faceNormalsPerVertex(_ m: Mesh3) -> [Vec3] {
        var n = [Vec3](repeating: Vec3(0, 0, 1), count: m.positions.count)
        var t = 0
        while t + 2 < m.indices.count {
            let a = Int(m.indices[t]), b = Int(m.indices[t + 1]), c = Int(m.indices[t + 2])
            let raw = cross3(m.positions[b] - m.positions[a], m.positions[c] - m.positions[a])
            if len3(raw) > 1e-300 { let fn = normalize3(raw); n[a] = fn; n[b] = fn; n[c] = fn }
            t += 3
        }
        return n
    }

    /// Triangulate a 3-D polygon in its own plane, keeping its winding. Returns local index triples.
    static func triangulate(_ p: [Vec3]) -> [(Int, Int, Int)] {
        let n = p.count
        if n < 3 { return [] }
        if n == 3 { return [(0, 1, 2)] }
        var nrm = Vec3(0, 0, 0)                                  // Newell normal
        for i in 0..<n {
            let a = p[i], b = p[(i + 1) % n]
            nrm += Vec3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y))
        }
        func fan() -> [(Int, Int, Int)] { (1..<(n - 1)).map { (0, $0, $0 + 1) } }
        guard len3(nrm) > 1e-300 else { return fan() }
        nrm = normalize3(nrm)
        let helper = abs(nrm.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0)
        let u = normalize3(cross3(helper, nrm)), v = cross3(nrm, u)   // (u, v, nrm) right-handed → CCW about nrm
        let flat = p.map { Vec2(dot3($0, u), dot3($0, v)) }
        let tris = Triangulation.earClip(flat)
        return tris.count == n - 2 ? tris : fan()
    }

    static func appendPolygon(_ idx: [UInt32], positions: [Vec3], to out: inout [UInt32]) {
        for (a, b, c) in triangulate(idx.map { positions[Int($0)] }) {
            out.append(idx[a]); out.append(idx[b]); out.append(idx[c])
        }
    }

    static func hasParallelNormals(_ m: Mesh3) -> Bool { m.normals.count == m.positions.count }
    static func hasParallelUVs(_ m: Mesh3) -> Bool { m.uvs.count == m.positions.count }
    static func utf8(_ s: String) -> Data { Data(s.utf8) }
}
