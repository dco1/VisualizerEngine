//  MeshIO+GPUWeld.swift — encodes the GPU weld + smooth pipeline (kernels: MeshIO+GPUKernelsWeld.swift) and the
//  per-triangle checks. Everything for one call goes into the caller's encoder; the real counts come back through the
//  `cnt` buffer after the command buffer has run (outputs are sized for the worst case: nothing is read mid-way).

#if canImport(Metal)
import Foundation
import Metal

extension MeshIOGPU {
    /// Buffers a weld pipeline leaves behind; `cnt[0]` = vertices kept, `cnt[1]` = triangles kept.
    struct Welded {
        let positions: MTLBuffer, normals: MTLBuffer, indices: MTLBuffer, cnt: MTLBuffer
    }

    /// Group vertices that agree on `round(p / q)`; `q` lives in a 1-float buffer. Returns the per-vertex canonical
    /// (lowest) index and the survivor flags. `rep`, `table` work buffers are returned only through these.
    func encodeGroups(_ e: MTLComputeCommandEncoder, P: MTLBuffer, n: Int, q: MTLBuffer) -> (canon: MTLBuffer, flag: MTLBuffer) {
        let size = pow2Table(for: n)
        let slots = alloc(size * 4), rep = alloc(n * 4), gmin = alloc(n * 4)
        let canon = alloc(n * 4), flag = alloc(n * 4)
        go(e, "fill_u32", size, [.buf(slots), .u(UInt32(size)), .u(0xFFFF_FFFF)])
        go(e, "fill_u32", n, [.buf(gmin), .u(UInt32(n)), .u(0xFFFF_FFFF)])
        go(e, "weld_insert", n, [.buf(P), .u(UInt32(n)), .buf(q), .buf(slots), .buf(rep), .u(UInt32(size - 1))])
        go(e, "weld_groupmin", n, [.buf(rep), .buf(gmin), .u(UInt32(n))])
        go(e, "weld_canon", n, [.buf(rep), .buf(gmin), .buf(canon), .buf(flag), .u(UInt32(n))])
        return (canon, flag)
    }

    /// Weld `P` (nV flat-float vertices) and its triangles `I` (nil = vertex 3t+k, STL's unshared layout), then smooth.
    /// `quantum`: nil = the CPU weld's rule (bounds diagonal · 1e-6, found on the GPU), else a fixed value.
    func encodeWeldSmooth(_ e: MTLComputeCommandEncoder, P: MTLBuffer, nV: Int, I: MTLBuffer?, nT: Int,
                          crease: Double, quantum: Float?) -> Welded {
        let cnt = alloc(64)
        let q = alloc(4)
        if let quantum { q.contents().storeBytes(of: quantum, as: Float.self) } else {
            let mm = alloc(24)
            let p = mm.contents().assumingMemoryBound(to: UInt32.self)
            for k in 0 ..< 3 { p[k] = 0xFFFF_FFFF; p[3 + k] = 0 }
            go(e, "bounds_minmax", nV, [.buf(P), .u(UInt32(nV)), .buf(mm)])
            go(e, "weld_quantum", 1, [.buf(mm), .buf(q)])
        }
        let (canon, flag) = encodeGroups(e, P: P, n: nV, q: q)
        let scanV = alloc(nV * 4), outP = alloc(nV * 12), remap = alloc(nV * 4)
        scan(e, flags: flag, out: scanV, n: nV, cnt: cnt, slot: 0)
        go(e, "weld_emit", nV, [.buf(P), .buf(canon), .buf(scanV), .buf(outP), .buf(remap), .u(UInt32(nV))])
        let flagT = alloc(nT * 4), ra = alloc(nT * 12), scanT = alloc(nT * 4), outI = alloc(nT * 12)
        go(e, "weld_tri_flags", nT, [.buf(I ?? flagT), .buf(remap), .buf(flagT), .buf(ra), .u(UInt32(nT)), .u(I == nil ? 1 : 0)])
        scan(e, flags: flagT, out: scanT, n: nT, cnt: cnt, slot: 1)
        go(e, "weld_tri_compact", nT, [.buf(ra), .buf(flagT), .buf(scanT), .buf(outI), .u(UInt32(nT))])
        let nrm = encodeSmoothNormals(e, P: outP, I: outI, cnt: cnt, slotV: 0, slotT: 1, nVMax: nV, nTMax: nT, crease: crease)
        return Welded(positions: outP, normals: nrm, indices: outI, cnt: cnt)
    }

    /// `Mesh3.smoothed`'s normals for an indexed mesh whose vertices are already shared (counts read from `cnt`).
    func encodeSmoothNormals(_ e: MTLComputeCommandEncoder, P: MTLBuffer, I: MTLBuffer, cnt: MTLBuffer, slotV: Int, slotT: Int,
                             nVMax: Int, nTMax: Int, crease: Double) -> MTLBuffer {
        let fn = alloc(nTMax * 12), last = alloc(nVMax * 4), acc = alloc(nVMax * 12), nrm = alloc(nVMax * 12)
        go(e, "sm_faces", nTMax, [.buf(P), .buf(I), .buf(cnt), .u(UInt32(slotT)), .buf(fn), .buf(last)])
        go(e, "sm_accum", nTMax, [.buf(P), .buf(I), .buf(cnt), .u(UInt32(slotT)), .buf(fn), .buf(last), .buf(acc),
                                  .f(Float(cos(crease * .pi / 180)))])
        go(e, "sm_finish", nVMax, [.buf(acc), .buf(fn), .buf(last), .buf(nrm), .buf(cnt), .u(UInt32(slotV))])
        return nrm
    }
}

/// What the GPU measured about a triangle mesh.
public struct MeshGPUReport: Sendable {
    /// Σ a·(b×c)/6 about the bounds' centre (translation-invariant for a closed mesh); > 0 for outward winding.
    public var signedVolume: Double
    public var area: Double
    public var degenerateTriangles: Int
    /// Triangles whose winding normal points against the mean of their vertex normals (0 if no normals were given).
    public var flippedAgainstNormals: Int
    /// A directed edge used twice: two triangles wound the same way across an edge (inconsistent winding / non-manifold).
    public var duplicateDirectedEdges: Int
    /// Directed edges with no opposite partner: boundary edges, or the other half of an inconsistent pair.
    public var openEdges: Int
    public var isClosedConsistent: Bool { duplicateDirectedEdges == 0 && openEdges == 0 && degenerateTriangles == 0 }
}

public extension MeshIO {
    /// Volume / winding checks on the GPU; nil without a Metal device or when an index is out of range.
    static func gpuReport(_ m: Mesh3) -> MeshGPUReport? {
        #if canImport(Metal)
        guard let g = MeshIOGPU.shared, !m.indices.isEmpty, m.indices.count % 3 == 0 else { return nil }
        let nV = m.positions.count, nT = m.indices.count / 3
        guard m.indices.allSatisfy({ Int($0) < nV }) else { return nil }
        let P = g.upload(MeshIOGPUUtil.narrow3(m.positions)), I = g.upload(m.indices)
        let hasN = m.normals.count == nV
        let N = hasN ? g.upload(MeshIOGPUUtil.narrow3(m.normals)) : g.alloc(16)
        let b = m.bounds!, o = (b.min + b.max) * 0.5
        let groups = (nT + 255) / 256
        let pf = g.alloc(groups * 16), pu = g.alloc(groups * 16)
        let cnt = g.alloc(16); cnt.contents().storeBytes(of: UInt32(nT), as: UInt32.self)
        let size = g.pow2Table(for: 3 * nT)
        let slots = g.alloc(size * 4), stats = g.alloc(16)
        let ok = g.run { e in
            g.go(e, "tri_check", groups * 256, [.buf(P), .buf(I), .buf(N), .buf(cnt), .u(0),
                                                MeshIOGPU.bytes(SIMD4<Float>(Float(o.x), Float(o.y), Float(o.z), 0)),
                                                .u(hasN ? 1 : 0), .buf(pf), .buf(pu)], group: 256)
            g.go(e, "fill_u32", size, [.buf(slots), .u(UInt32(size)), .u(0xFFFF_FFFF)])
            g.go(e, "edge_insert", 3 * nT, [.buf(I), .buf(cnt), .u(0), .buf(slots), .u(UInt32(size - 1)), .buf(stats)])
            g.go(e, "edge_reverse", 3 * nT, [.buf(I), .buf(cnt), .u(0), .buf(slots), .u(UInt32(size - 1)), .buf(stats)])
        }
        guard ok else { return nil }
        let f = MeshIOGPU.read(pf, SIMD4<Float>.self, count: groups), u = MeshIOGPU.read(pu, SIMD4<UInt32>.self, count: groups)
        let s = MeshIOGPU.read(stats, UInt32.self, count: 4)
        return MeshGPUReport(signedVolume: f.reduce(0.0) { $0 + Double($1.x) }, area: f.reduce(0.0) { $0 + Double($1.y) },
                             degenerateTriangles: u.reduce(0) { $0 + Int($1.x) } , flippedAgainstNormals: u.reduce(0) { $0 + Int($1.y) },
                             duplicateDirectedEdges: Int(s[0]), openEdges: Int(s[1]))
        #else
        return nil
        #endif
    }
}
#endif
