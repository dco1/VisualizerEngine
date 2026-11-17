//  MeshIO+GPUPLY.swift — binary PLY on the GPU. Read: the header is parsed here (CPU, a few hundred bytes), the vertex
//  and face bodies by `ply_vertices` / `ply_faces`. Anything the fixed-stride kernels can't promise returns nil and the
//  CPU reader runs: ascii, n-gon faces, other list properties, float64 or 32-bit-integer positions (a float loses them),
//  faces before vertices, a body shorter than the header says (the CPU reader owns the error). Files without normals get
//  `Mesh3.smoothed`'s normals on the GPU when no two vertices share a 0.1 mm cell (the CPU smooths ACROSS such twins,
//  by position), else the smoothing runs on the CPU over the GPU-decoded mesh. Write: binary LE, 8 floats (+ 3 uchar)
//  a vertex and `3 i32 i32 i32` a face, as the CPU writer.

import Foundation

#if canImport(Metal)
import Metal

extension MeshIO {
    private struct PProp { var name: String; var type: Int; var list: Int? }          // type codes as in the kernel
    private struct PElem { var name: String; var count: Int; var props: [PProp] = [] }
    private static let typeCodes: [String: Int] = [
        "char": 0, "int8": 0, "uchar": 1, "uint8": 1, "short": 2, "int16": 2, "ushort": 3, "uint16": 3,
        "int": 4, "int32": 4, "uint": 5, "uint32": 5, "float": 6, "float32": 6, "double": 7, "float64": 7]
    private static let typeSize = [1, 1, 2, 2, 4, 4, 4, 8]

    static func gpuReadPLY(_ data: Data) -> Result? {
        guard let g = MeshIOGPU.shared, data.starts(with: Array("ply".utf8)),
              let hr = data.range(of: Data("end_header".utf8)) else { return nil }
        var body = hr.upperBound
        while body < data.count, data[body] != 10 { body += 1 }
        body += 1
        guard body <= data.count, body < (1 << 31), data.count < (1 << 31) else { return nil }
        var be = false, elems: [PElem] = []
        for line in String(decoding: data[0 ..< hr.lowerBound], as: UTF8.self).split(whereSeparator: \.isNewline) {
            let t = line.split(separator: " ")
            guard let k = t.first else { continue }
            if k == "format" {
                guard t.count >= 2 else { return nil }
                if t[1] == "binary_big_endian" { be = true } else if t[1] != "binary_little_endian" { return nil }
            } else if k == "element" {
                guard t.count == 3, let n = Int(t[2]), n >= 0 else { return nil }
                elems.append(PElem(name: String(t[1]), count: n))
            } else if k == "property", !elems.isEmpty {
                if t.count == 5, t[1] == "list", let c = typeCodes[String(t[2])], let i = typeCodes[String(t[3])] {
                    elems[elems.count - 1].props.append(PProp(name: String(t[4]), type: i, list: c))
                } else if t.count == 3, let ty = typeCodes[String(t[1])] {
                    elems[elems.count - 1].props.append(PProp(name: String(t[2]), type: ty, list: nil))
                } else { return nil }
            }
        }
        // Lay the elements out; only a vertex element of scalars and a triangle face element are fixed-stride.
        var offset = body
        var vInfo: (base: Int, stride: Int, count: Int, props: [PProp])?
        var fInfo: (base: Int, stride: Int, count: Int, listOff: Int, cntType: Int, idxType: Int)?
        for el in elems {
            guard el.count < 100_000_000 else { return nil }
            let lists = el.props.filter { $0.list != nil }
            var stride = 0
            if lists.isEmpty {
                stride = el.props.reduce(0) { $0 + typeSize[$1.type] }
                if el.name == "vertex" { guard vInfo == nil else { return nil }; vInfo = (offset, stride, el.count, el.props) }
            } else {
                guard el.name == "face", lists.count == 1, let l = lists.first, (l.name == "vertex_indices" || l.name == "vertex_index"),
                      l.type <= 5, l.list! <= 5, fInfo == nil, vInfo != nil else { return nil }
                var before = 0, found = false
                stride = 0
                for p in el.props {
                    if p.list != nil { found = true; stride += typeSize[p.list!] + 3 * typeSize[p.type] }
                    else { stride += typeSize[p.type]; if !found { before += typeSize[p.type] } }
                }
                fInfo = (offset, stride, el.count, before, l.list!, l.type)
            }
            offset += stride * el.count
        }
        guard let v = vInfo, let f = fInfo, f.count > 0, v.count > 0, offset <= data.count else { return nil }

        // channel -> (offset, type): x y z nx ny nz u v r g b
        var map = [Int32](repeating: -1, count: 11), typ = [UInt32](repeating: 0, count: 11)
        var at = 0, offs: [String: (Int, Int)] = [:]
        for p in v.props { offs[p.name] = (at, p.type); at += typeSize[p.type] }
        func pick(_ names: [String]) -> (Int, Int)? { names.compactMap { offs[$0] }.first }
        let chans: [(Int, [String])] = [(0, ["x"]), (1, ["y"]), (2, ["z"]), (3, ["nx"]), (4, ["ny"]), (5, ["nz"]),
            (6, ["s", "u", "texture_u", "texture_s"]), (7, ["t", "v", "texture_v", "texture_t"]),
            (8, ["red"]), (9, ["green"]), (10, ["blue"])]
        for (c, names) in chans { if let (o, t) = pick(names) { map[c] = Int32(o); typ[c] = UInt32(t) } }
        guard map[0] >= 0, map[1] >= 0, map[2] >= 0 else { return nil }
        for c in 0 ..< 3 where !(typ[c] == 6 || typ[c] < 4) { return nil }            // positions: float32 / 8, 16-bit ints
        let hasN = map[3] >= 0 && map[4] >= 0 && map[5] >= 0, hasC = map[8] >= 0 && map[9] >= 0 && map[10] >= 0
        guard !(hasN) || true else { return nil }

        struct PlyMap { var off: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
                        var typ: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
                        var be: UInt32, hasN: UInt32, hasC: UInt32, pad: UInt32 }
        let pm = PlyMap(off: (map[0], map[1], map[2], map[3], map[4], map[5], map[6], map[7], map[8], map[9], map[10]),
                        typ: (typ[0], typ[1], typ[2], typ[3], typ[4], typ[5], typ[6], typ[7], typ[8], typ[9], typ[10]),
                        be: be ? 1 : 0, hasN: hasN ? 1 : 0, hasC: hasC ? 1 : 0, pad: 0)
        let nV = v.count, nT = f.count
        let raw = g.upload(data)
        let P = g.alloc(nV * 12), N = g.alloc(nV * 12), U = g.alloc(nV * 8), C = g.alloc(hasC ? nV * 12 : 16)
        let I = g.alloc(nT * 12), bad = g.alloc(16), cnt = g.alloc(64)
        cnt.contents().storeBytes(of: UInt32(nV), toByteOffset: 0, as: UInt32.self)
        cnt.contents().storeBytes(of: UInt32(nT), toByteOffset: 4, as: UInt32.self)
        var smoothed: MTLBuffer?, dupFlags: MTLBuffer?
        guard g.run({ e in
            g.go(e, "ply_vertices", nV, [.buf(raw), .buf(P), .buf(N), .buf(U), .buf(C),
                                         MeshIOGPU.bytes(array: [UInt32(v.base), UInt32(v.stride), UInt32(nV), 0]), MeshIOGPU.bytes(pm)])
            g.go(e, "ply_faces", nT, [.buf(raw), .buf(I),
                                      MeshIOGPU.bytes(array: [UInt32(f.base), UInt32(f.stride), UInt32(nT), UInt32(f.listOff),
                                                       UInt32(f.cntType), UInt32(f.idxType), be ? 1 : 0, UInt32(nV)]), .buf(bad)])
            if !hasN {
                // Speculative: smooth by index, and count how many vertices a 0.1 mm weld would merge.
                smoothed = g.encodeSmoothNormals(e, P: P, I: I, cnt: cnt, slotV: 0, slotT: 1, nVMax: nV, nTMax: nT, crease: 40)
                let q = g.alloc(4); q.contents().storeBytes(of: Float(1e-4), as: Float.self)
                let (_, flag) = g.encodeGroups(e, P: P, n: nV, q: q)
                dupFlags = g.alloc(nV * 4)
                g.scan(e, flags: flag, out: dupFlags!, n: nV, cnt: cnt, slot: 2)
            }
        }) else { return nil }
        let b = MeshIOGPU.read(bad, UInt32.self, count: 2)
        guard b[0] == 0, b[1] == 0 else { return nil }
        var fm = MeshIOFloatMesh()
        fm.positions = MeshIOGPU.read(P, Float.self, count: nV * 3)
        fm.uvs = MeshIOGPU.read(U, Float.self, count: nV * 2)
        fm.indices = MeshIOGPU.read(I, UInt32.self, count: nT * 3)
        if hasC { fm.colors = MeshIOGPU.read(C, Float.self, count: nV * 3) }
        var mesh: Mesh3
        if hasN {
            fm.normals = MeshIOGPU.read(N, Float.self, count: nV * 3)
            mesh = fm.toMesh3()
        } else if MeshIOGPU.read(cnt, UInt32.self, count: 3)[2] == UInt32(nV), let s = smoothed {
            fm.normals = MeshIOGPU.read(s, Float.self, count: nV * 3)
            mesh = fm.toMesh3()
        } else {
            mesh = smoothedFromFaces(fm.toMesh3(), crease: 40)
        }
        var r = Result()
        r.meshes = [mesh]; r.names = [""]; r.materialNames = [nil]; r.colors = [hasC ? fm.colors3 : []]
        return r
    }

    static func gpuWritePLY(_ meshes: [Mesh3], options: WriteOptions) -> Data? {
        guard let g = MeshIOGPU.shared, !meshes.isEmpty else { return nil }
        let withN = meshes.filter { $0.normals.count == $0.positions.count }.count
        guard withN == 0 || withN == meshes.count else { return nil }                // mixed: CPU
        var m = Mesh3(); var cols: [SIMD3<Float>] = []; var haveC = true
        var uv: [Float] = []
        for (i, x) in meshes.enumerated() {
            var mm = x
            if withN == 0 { mm.normals = [] }
            let u = x.uvs.count == x.positions.count ? MeshIOGPUUtil.narrow2(x.uvs) : [Float](repeating: 0, count: 2 * x.positions.count)
            uv += u
            mm.uvs = []
            m.positions += mm.positions; m.normals += mm.normals
            let base = UInt32(m.positions.count - x.positions.count)
            m.indices += x.indices.map { $0 + base }
            if i < options.colors.count, options.colors[i].count == x.positions.count { cols += options.colors[i] } else { haveC = false }
        }
        let nV = m.positions.count, nT = m.indices.count / 3
        guard nV > 0, nT > 0, m.indices.allSatisfy({ Int($0) < nV }) else { return nil }
        var head = "ply\nformat binary_little_endian 1.0\ncomment VisualizerMaterials MeshIO\n"
        head += "element vertex \(nV)\n"
        for p in ["x", "y", "z", "nx", "ny", "nz", "s", "t"] { head += "property float \(p)\n" }
        if haveC { for p in ["red", "green", "blue"] { head += "property uchar \(p)\n" } }
        head += "element face \(nT)\nproperty list uchar int vertex_indices\nend_header\n"
        let hb = Array(head.utf8), vStride = haveC ? 35 : 32
        let total = hb.count + vStride * nV + 13 * nT
        let page = Int(getpagesize()), cap = (total + page - 1) / page * page
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, page, cap) == 0, let mem = raw else { return nil }
        hb.withUnsafeBytes { mem.copyMemory(from: $0.baseAddress!, byteCount: hb.count) }
        guard let out = g.device.makeBuffer(bytesNoCopy: mem, length: cap, options: .storageModeShared, deallocator: nil) else { free(mem); return nil }
        let P = g.upload(MeshIOGPUUtil.narrow3(m.positions)), I = g.upload(m.indices), U = g.upload(uv)
        let Cb = g.upload(haveC ? cols.flatMap { [$0.x, $0.y, $0.z] } : [Float]())
        let N = withN == 0 ? g.alloc(nV * 12) : g.upload(MeshIOGPUUtil.narrow3(m.normals))
        let cnt = g.alloc(16)
        cnt.contents().storeBytes(of: UInt32(nT), toByteOffset: 4, as: UInt32.self)
        let ok = g.run { e in
            if withN == 0 {                                    // faceNormalsPerVertex: the last face to touch a vertex
                let fn = g.alloc(nT * 12), last = g.alloc(nV * 4)
                g.go(e, "sm_faces", nT, [.buf(P), .buf(I), .buf(cnt), .u(1), .buf(fn), .buf(last)])
                g.go(e, "nrm_from_last", nV, [.buf(fn), .buf(last), .buf(N), .u(UInt32(nV))])
            }
            g.go(e, "ply_write_vertices", nV, [.buf(P), .buf(N), .buf(U), .buf(Cb), .buf(out), .u(UInt32(nV)), .u(UInt32(hb.count)), .u(haveC ? 1 : 0)])
            g.go(e, "ply_write_faces", nT, [.buf(I), .buf(out), .u(UInt32(nT)), .u(UInt32(hb.count + vStride * nV))])
        }
        guard ok else { free(mem); return nil }
        return Data(bytesNoCopy: mem, count: total, deallocator: .free)
    }
}
#endif
