//  HalfEdgeMesh+GPUTopology.swift — the directed-edge table and the Mesh3 → HalfEdgeMesh weld on the GPU (ticket item M4).
//
//  Directed-edge table: every halfedge inserts (origin, destination) into an open-addressing hash table with a
//  compare-and-swap (the table stores halfedge ids; a key is read back from the SoA, so no 64-bit atomics are needed), then
//  every halfedge looks up its reverse key = its twin. Edge ids come from a prefix scan over "canonical" halfedges
//  (no twin, or lower id than its twin), which numbers edges exactly as the CPU's `addFace` order does.
//
//  `HalfEdgeMesh.importedGPU` is `imported(_:)` end to end on the GPU in ONE command buffer, no sync in between:
//    weld         corner positions quantised to `weldTolerance`, hash-inserted with atomic-min, so each corner learns the
//                 lowest corner id sharing its point; flag + scan numbers the welded vertices in first-occurrence order
//                 (the CPU's order); degenerate triangles (a welded corner repeated, or ~zero area) are flagged out
//    compact      scan of the valid triangles → face ids; halfedges 3f + k (next / prev / face / origin) written
//    twins/edges  the directed-edge table above; vertex → halfedge by atomic-min (the lowest halfedge, as the CPU's first)
//  and then the CPU only slices the buffers to the real counts and gathers Mesh3's per-corner uv / normal (a memcpy-like
//  gather) and refills `directed`, the dictionary the mutation primitives use (a stored property of HalfEdgeMesh: it is a
//  CPU hash map, so it is the part of an import the GPU cannot take over without changing HalfEdgeMesh.swift).
//  Guarantees: on a mesh with no duplicate directed edge the result equals `imported(_:)` array for array (tested).
//  Not guaranteed / falls back to the CPU (returns nil): a duplicate directed edge anywhere (CPU retries each such
//  triangle reversed, in order — sequential), no device. The weld runs in Float: points that are exactly equal in
//  Float weld, points within `weldTolerance` of each other but on different Float values may weld differently from the
//  CPU's Double quantisation.

import Foundation
import simd

public struct HalfEdgeGPUTable: Sendable {
    public var twin: [Int32], edge: [Int32], edgeHalfedge: [Int32]
    /// Halfedges whose directed edge was already taken (non-manifold input).
    public var duplicates: Int
}

public extension HalfEdgeMesh {
    /// Twin and edge ids for flat halfedge arrays (dead slots: `face < 0`), built on the GPU. nil without a device.
    static func gpuEdgeTable(origin: [Int32], next: [Int32], face: [Int32]) -> HalfEdgeGPUTable? {
        HalfEdgeTopologyGPU.table(origin: origin, next: next, face: face)
    }

    /// `imported(_:)` on the GPU, or nil (see the file header) — callers fall back to `imported`.
    static func importedGPU(_ mesh: Mesh3, weldTolerance: Double = 1e-6) -> (mesh: HalfEdgeMesh, report: ImportReport)? {
        guard let r = HalfEdgeTopologyGPU.weldAndBuild(mesh, tolerance: weldTolerance), r.duplicates == 0 else { return nil }
        var m = HalfEdgeMesh()
        let nt = mesh.indices.count / 3, nf = r.faceCount, nh = nf * 3
        let hasN = mesh.normals.count == mesh.positions.count, hasUV = mesh.uvs.count == mesh.positions.count
        m.positions = r.repCorner.map { mesh.positions[Int(mesh.indices[Int($0)])] }
        m.vertexHE = r.vertexHalfedge.map { $0 == Int32.max ? -1 : Int($0) }
        m.heOrigin = r.origin[0..<nh].map { Int($0) }
        m.heNext = r.next[0..<nh].map { Int($0) }
        m.hePrev = r.prev[0..<nh].map { Int($0) }
        m.heTwin = r.twin[0..<nh].map { Int($0) }
        m.heFace = r.face[0..<nh].map { Int($0) }
        m.heEdge = r.edge[0..<nh].map { Int($0) }
        m.faceHE = (0..<nf).map { $0 * 3 }
        m.edgeHE = r.edgeHalfedge[0..<r.edgeCount].map { Int($0) }
        m.heUV = [Vec2](repeating: .zero, count: nh); m.heNormal = [Vec3](repeating: .zero, count: nh)
        for f in 0..<nf {
            let t = Int(r.sourceTriangle[f])
            var flat = Vec3.zero
            if !hasN { flat = normalize3(HalfEdgeMesh.newell((0..<3).map { mesh.positions[Int(mesh.indices[t * 3 + $0])] })) }
            for k in 0..<3 {
                let i = Int(mesh.indices[t * 3 + k])
                m.heUV[f * 3 + k] = hasUV ? mesh.uvs[i] : .zero
                m.heNormal[f * 3 + k] = hasN ? mesh.normals[i] : flat
            }
        }
        m.directed.reserveCapacity(nh)
        for h in 0..<nh { m.directed[HalfEdgeMesh.key(m.heOrigin[h], m.heOrigin[m.heNext[h]])] = h }
        return (m, ImportReport(degenerate: nt - nf, reversed: 0, rejected: 0))
    }

    /// GPU import when it applies, else the CPU's.
    static func importedAuto(_ mesh: Mesh3, weldTolerance: Double = 1e-6) -> (mesh: HalfEdgeMesh, report: ImportReport) {
        importedGPU(mesh, weldTolerance: weldTolerance) ?? imported(mesh, weldTolerance: weldTolerance)
    }
}

struct HalfEdgeGPUImport {
    var faceCount = 0, edgeCount = 0, duplicates = 0
    var repCorner: [Int32] = [], sourceTriangle: [Int32] = [], vertexHalfedge: [Int32] = []
    var origin: [Int32] = [], next: [Int32] = [], prev: [Int32] = [], face: [Int32] = []
    var twin: [Int32] = [], edge: [Int32] = [], edgeHalfedge: [Int32] = []
}

#if canImport(Metal)
import Metal

enum HalfEdgeTopologyGPU {
    static func pow2(atLeast n: Int) -> Int { var p = 16; while p < n { p <<= 1 }; return p }

    /// Encode: hash table → twins → canonical flags → edge ids. Buffers are sized for `h` halfedge slots.
    static func encodeTable(_ g: ModellingGPU, _ enc: MTLComputeCommandEncoder, h: Int, org: MTLBuffer, nxt: MTLBuffer, face: MTLBuffer,
                            twin: MTLBuffer, edge: MTLBuffer, edgeHE: MTLBuffer, eidx: MTLBuffer, dup: MTLBuffer) {
        let size = pow2(atLeast: h * 2)
        let table = g.buffer(length: size * 4), flag = g.buffer(length: (h + 1) * 4)
        g.fill(enc, table, count: size, value: 0xFFFF_FFFF)
        g.dispatch1D(enc, "ht_insert", h, [.buf(org), .buf(nxt), .buf(face), .buf(table), .buf(dup), .value(UInt32(h)), .value(UInt32(size - 1))])
        g.dispatch1D(enc, "ht_twins", h, [.buf(org), .buf(nxt), .buf(face), .buf(table), .buf(twin), .value(UInt32(h)), .value(UInt32(size - 1))])
        g.fill(enc, flag, count: h + 1, value: 0)
        g.dispatch1D(enc, "ht_canon", h, [.buf(twin), .buf(face), .buf(flag), .value(UInt32(h))])
        g.scan(enc, input: flag, output: eidx, n: h + 1)
        g.dispatch1D(enc, "ht_edges", h, [.buf(twin), .buf(face), .buf(eidx), .buf(edge), .buf(edgeHE), .value(UInt32(h))])
    }

    static func table(origin: [Int32], next: [Int32], face: [Int32]) -> HalfEdgeGPUTable? {
        guard let g = ModellingGPU.shared, !origin.isEmpty else { return nil }
        let h = origin.count
        let org = g.buffer(origin), nxt = g.buffer(next), fc = g.buffer(face)
        let twin = g.buffer(length: h * 4), edge = g.buffer(length: h * 4), edgeHE = g.buffer(length: h * 4)
        let eidx = g.buffer(length: (h + 1) * 4), dup = g.buffer([UInt32(0)])
        guard g.run({ encodeTable(g, $0, h: h, org: org, nxt: nxt, face: fc, twin: twin, edge: edge, edgeHE: edgeHE, eidx: eidx, dup: dup) })
        else { return nil }
        let ne = Int(g.read(eidx, UInt32.self, count: h + 1)[h])
        return HalfEdgeGPUTable(twin: g.read(twin, Int32.self, count: h), edge: g.read(edge, Int32.self, count: h),
                                edgeHalfedge: Array(g.read(edgeHE, Int32.self, count: h).prefix(ne)),
                                duplicates: Int(dup.contents().assumingMemoryBound(to: UInt32.self)[0]))
    }

    static func weldAndBuild(_ mesh: Mesh3, tolerance: Double) -> HalfEdgeGPUImport? {
        guard let g = ModellingGPU.shared else { return nil }
        let nt = mesh.indices.count / 3, nc = nt * 3
        guard nt > 0 else { return nil }
        let hmax = nc
        let pos = g.buffer(mesh.positions.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), 0) })
        let ind = g.buffer(Array(mesh.indices.prefix(nc)))
        let wsize = pow2(atLeast: nc * 2)
        let wtable = g.buffer(length: wsize * 4)
        let rep = g.buffer(length: nc * 4), isRep = g.buffer(length: (nc + 1) * 4), vid = g.buffer(length: (nc + 1) * 4)
        let cornerV = g.buffer(length: nc * 4), repList = g.buffer(length: nc * 4)
        let valid = g.buffer(length: (nt + 1) * 4), fidx = g.buffer(length: (nt + 1) * 4)
        let org = g.buffer(length: hmax * 4), nxt = g.buffer(length: hmax * 4), prv = g.buffer(length: hmax * 4)
        let face = g.buffer(length: hmax * 4), src = g.buffer(length: nt * 4)
        let twin = g.buffer(length: hmax * 4), edge = g.buffer(length: hmax * 4), edgeHE = g.buffer(length: hmax * 4)
        let eidx = g.buffer(length: (hmax + 1) * 4), dup = g.buffer([UInt32(0)])
        let vhe = g.buffer(length: nc * 4)
        let inv = Float(1 / Swift.max(tolerance, 1e-12))
        let ok = g.run { enc in
            g.fill(enc, wtable, count: wsize, value: 0xFFFF_FFFF)
            g.dispatch1D(enc, "wd_insert", nc, [.buf(pos), .buf(ind), .buf(wtable), .value(UInt32(nc)), .value(UInt32(wsize - 1)), .value(inv)])
            g.dispatch1D(enc, "wd_resolve", nc, [.buf(pos), .buf(ind), .buf(wtable), .buf(rep), .value(UInt32(nc)), .value(UInt32(wsize - 1)), .value(inv)])
            g.fill(enc, isRep, count: nc + 1, value: 0)
            g.dispatch1D(enc, "wd_flags", nc, [.buf(rep), .buf(isRep), .value(UInt32(nc))])
            g.scan(enc, input: isRep, output: vid, n: nc + 1)
            g.dispatch1D(enc, "wd_assign", nc, [.buf(rep), .buf(vid), .buf(cornerV), .buf(repList), .value(UInt32(nc))])
            g.fill(enc, valid, count: nt + 1, value: 0)
            g.dispatch1D(enc, "wd_valid", nt, [.buf(pos), .buf(ind), .buf(cornerV), .buf(valid), .value(UInt32(nt))])
            g.scan(enc, input: valid, output: fidx, n: nt + 1)
            g.fill(enc, face, count: hmax, value: 0xFFFF_FFFF)
            g.dispatch1D(enc, "wd_compact", nt, [.buf(cornerV), .buf(valid), .buf(fidx), .buf(org), .buf(nxt), .buf(prv), .buf(face), .buf(src), .value(UInt32(nt))])
            g.fill(enc, vhe, count: nc, value: 0xFFFF_FFFF)
            g.dispatch1D(enc, "wd_vertex_he", hmax, [.buf(org), .buf(face), .buf(vhe), .value(UInt32(hmax))])
            encodeTable(g, enc, h: hmax, org: org, nxt: nxt, face: face, twin: twin, edge: edge, edgeHE: edgeHE, eidx: eidx, dup: dup)
        }
        guard ok else { return nil }
        var r = HalfEdgeGPUImport()
        let nv = Int(g.read(vid, UInt32.self, count: nc + 1)[nc])
        r.faceCount = Int(g.read(fidx, UInt32.self, count: nt + 1)[nt])
        r.edgeCount = Int(g.read(eidx, UInt32.self, count: hmax + 1)[hmax])
        r.duplicates = Int(dup.contents().assumingMemoryBound(to: UInt32.self)[0])
        r.repCorner = Array(g.read(repList, Int32.self, count: nc).prefix(nv))
        let nh = r.faceCount * 3
        r.sourceTriangle = Array(g.read(src, Int32.self, count: nt).prefix(r.faceCount))
        r.vertexHalfedge = Array(g.read(vhe, UInt32.self, count: nc).prefix(nv)).map { $0 == 0xFFFF_FFFF ? Int32.max : Int32($0) }
        r.origin = Array(g.read(org, Int32.self, count: hmax).prefix(nh)); r.next = Array(g.read(nxt, Int32.self, count: hmax).prefix(nh))
        r.prev = Array(g.read(prv, Int32.self, count: hmax).prefix(nh)); r.face = Array(g.read(face, Int32.self, count: hmax).prefix(nh))
        r.twin = Array(g.read(twin, Int32.self, count: hmax).prefix(nh)); r.edge = Array(g.read(edge, Int32.self, count: hmax).prefix(nh))
        r.edgeHalfedge = Array(g.read(edgeHE, Int32.self, count: hmax).prefix(r.edgeCount))
        return r
    }

    static let source = """
    static inline uint ht_hash(uint a, uint b) {
        uint x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
        x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
        return x;
    }

    // Insert every alive halfedge under key (origin, destination); a taken key is a duplicate directed edge.
    kernel void ht_insert(device const int* org [[buffer(0)]], device const int* nxt [[buffer(1)]], device const int* face [[buffer(2)]],
                          device atomic_uint* table [[buffer(3)]], device atomic_uint* dup [[buffer(4)]], constant uint& n [[buffer(5)]],
                          constant uint& mask [[buffer(6)]], uint h [[thread_position_in_grid]]) {
        if (h >= n || face[h] < 0) return;
        uint a = uint(org[h]), b = uint(org[nxt[h]]);
        uint s = ht_hash(a, b) & mask;
        for (uint probe = 0; probe <= mask; probe++) {
            uint cur = 0xFFFFFFFFu;
            if (atomic_compare_exchange_weak_explicit(&table[s], &cur, h, memory_order_relaxed, memory_order_relaxed)) return;
            if (cur == 0xFFFFFFFFu) continue;                          // spurious failure: retry this slot
            if (uint(org[cur]) == a && uint(org[nxt[cur]]) == b) { atomic_fetch_add_explicit(dup, 1u, memory_order_relaxed); return; }
            s = (s + 1) & mask; probe = probe;
        }
    }

    kernel void ht_twins(device const int* org [[buffer(0)]], device const int* nxt [[buffer(1)]], device const int* face [[buffer(2)]],
                         device atomic_uint* table [[buffer(3)]], device int* twin [[buffer(4)]], constant uint& n [[buffer(5)]],
                         constant uint& mask [[buffer(6)]], uint h [[thread_position_in_grid]]) {
        if (h >= n) return;
        twin[h] = -1;
        if (face[h] < 0) return;
        uint a = uint(org[h]), b = uint(org[nxt[h]]);
        uint s = ht_hash(b, a) & mask;
        for (uint probe = 0; probe <= mask; probe++) {
            uint cur = atomic_load_explicit(&table[s], memory_order_relaxed);
            if (cur == 0xFFFFFFFFu) return;
            if (uint(org[cur]) == b && uint(org[nxt[cur]]) == a) { twin[h] = int(cur); return; }
            s = (s + 1) & mask;
        }
    }

    kernel void ht_canon(device const int* twin [[buffer(0)]], device const int* face [[buffer(1)]], device uint* flag [[buffer(2)]],
                         constant uint& n [[buffer(3)]], uint h [[thread_position_in_grid]]) {
        if (h >= n) return;
        flag[h] = (face[h] >= 0 && (twin[h] < 0 || int(h) < twin[h])) ? 1 : 0;
    }

    kernel void ht_edges(device const int* twin [[buffer(0)]], device const int* face [[buffer(1)]], device const uint* eidx [[buffer(2)]],
                         device int* edge [[buffer(3)]], device int* edgeHE [[buffer(4)]], constant uint& n [[buffer(5)]],
                         uint h [[thread_position_in_grid]]) {
        if (h >= n) return;
        if (face[h] < 0) { edge[h] = -1; return; }
        bool canon = twin[h] < 0 || int(h) < twin[h];
        edge[h] = int(canon ? eidx[h] : eidx[uint(twin[h])]);
        if (canon) edgeHE[eidx[h]] = int(h);
    }

    // ---- weld ----
    static inline int3 wd_q(float4 p, float inv) { return int3(rint(p.xyz * inv)); }
    static inline uint wd_hash(int3 q) { return ht_hash(uint(q.x) * 0x9E3779B1u ^ uint(q.y), uint(q.z)); }

    kernel void wd_insert(device const float4* pos [[buffer(0)]], device const uint* ind [[buffer(1)]], device atomic_uint* table [[buffer(2)]],
                          constant uint& n [[buffer(3)]], constant uint& mask [[buffer(4)]], constant float& inv [[buffer(5)]],
                          uint c [[thread_position_in_grid]]) {
        if (c >= n) return;
        int3 q = wd_q(pos[ind[c]], inv);
        uint s = wd_hash(q) & mask;
        for (uint probe = 0; probe <= mask; probe++) {
            uint cur = 0xFFFFFFFFu;
            if (atomic_compare_exchange_weak_explicit(&table[s], &cur, c, memory_order_relaxed, memory_order_relaxed)) return;
            if (cur == 0xFFFFFFFFu) continue;
            if (all(wd_q(pos[ind[cur]], inv) == q)) { atomic_fetch_min_explicit(&table[s], c, memory_order_relaxed); return; }
            s = (s + 1) & mask;
        }
    }

    kernel void wd_resolve(device const float4* pos [[buffer(0)]], device const uint* ind [[buffer(1)]], device const uint* table [[buffer(2)]],
                           device uint* rep [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& mask [[buffer(5)]],
                           constant float& inv [[buffer(6)]], uint c [[thread_position_in_grid]]) {
        if (c >= n) return;
        int3 q = wd_q(pos[ind[c]], inv);
        uint s = wd_hash(q) & mask;
        for (uint probe = 0; probe <= mask; probe++) {
            uint cur = table[s];
            if (cur != 0xFFFFFFFFu && all(wd_q(pos[ind[cur]], inv) == q)) { rep[c] = cur; return; }
            s = (s + 1) & mask;
        }
        rep[c] = c;
    }

    kernel void wd_flags(device const uint* rep [[buffer(0)]], device uint* isRep [[buffer(1)]], constant uint& n [[buffer(2)]],
                         uint c [[thread_position_in_grid]]) {
        if (c < n) isRep[c] = rep[c] == c ? 1 : 0;
    }

    kernel void wd_assign(device const uint* rep [[buffer(0)]], device const uint* vid [[buffer(1)]], device uint* cornerV [[buffer(2)]],
                          device uint* repList [[buffer(3)]], constant uint& n [[buffer(4)]], uint c [[thread_position_in_grid]]) {
        if (c >= n) return;
        cornerV[c] = vid[rep[c]];
        if (rep[c] == c) repList[vid[c]] = c;
    }

    // A triangle is kept when its three welded vertices differ and its area vector isn't ~0 (the CPU's 1e-14 on the Newell length).
    kernel void wd_valid(device const float4* pos [[buffer(0)]], device const uint* ind [[buffer(1)]], device const uint* cornerV [[buffer(2)]],
                         device uint* valid [[buffer(3)]], constant uint& nt [[buffer(4)]], uint t [[thread_position_in_grid]]) {
        if (t >= nt) return;
        uint a = cornerV[3*t], b = cornerV[3*t+1], c = cornerV[3*t+2];
        float3 p0 = pos[ind[3*t]].xyz, p1 = pos[ind[3*t+1]].xyz, p2 = pos[ind[3*t+2]].xyz;
        bool ok = a != b && b != c && a != c && length(cross(p1 - p0, p2 - p0)) >= 1e-14f;
        valid[t] = ok ? 1 : 0;
    }

    kernel void wd_compact(device const uint* cornerV [[buffer(0)]], device const uint* valid [[buffer(1)]], device const uint* fidx [[buffer(2)]],
                           device int* org [[buffer(3)]], device int* nxt [[buffer(4)]], device int* prv [[buffer(5)]],
                           device int* face [[buffer(6)]], device int* src [[buffer(7)]], constant uint& nt [[buffer(8)]],
                           uint t [[thread_position_in_grid]]) {
        if (t >= nt || valid[t] == 0) return;
        uint f = fidx[t];
        src[f] = int(t);
        for (uint k = 0; k < 3; k++) {
            uint h = 3 * f + k;
            org[h] = int(cornerV[3*t + k]); nxt[h] = int(3 * f + (k + 1) % 3); prv[h] = int(3 * f + (k + 2) % 3); face[h] = int(f);
        }
    }

    kernel void wd_vertex_he(device const int* org [[buffer(0)]], device const int* face [[buffer(1)]], device atomic_uint* vhe [[buffer(2)]],
                             constant uint& n [[buffer(3)]], uint h [[thread_position_in_grid]]) {
        if (h >= n || face[h] < 0) return;
        atomic_fetch_min_explicit(&vhe[uint(org[h])], h, memory_order_relaxed);
    }
    """
}
#else
enum HalfEdgeTopologyGPU {
    static func table(origin: [Int32], next: [Int32], face: [Int32]) -> HalfEdgeGPUTable? { nil }
    static func weldAndBuild(_ mesh: Mesh3, tolerance: Double) -> HalfEdgeGPUImport? { nil }
}
#endif
