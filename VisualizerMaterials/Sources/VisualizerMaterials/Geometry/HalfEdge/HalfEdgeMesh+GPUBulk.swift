//  HalfEdgeMesh+GPUBulk.swift — the half-edge mesh's bulk per-element maths on the GPU (ticket item M4).
//
//  `gpuBulk()` uploads the flat SoA topology once and, in ONE command buffer with no CPU step between, runs
//    face Newell vectors → { face normals, face areas }, then per-vertex area-weighted normals (each vertex walks its
//    own fan), then the signed volume (fan triangles about the mesh's centre, per-threadgroup partial sums, then a
//    one-group reduction). `cpuBulk()` is the Double-precision twin: oracle and fallback.
//  Guarantees: GPU agrees with the CPU to Float tolerance (~1e-5 relative per face / vertex; the volume sums in Float
//  about the centre, ~1e-4 relative). Dead faces give a zero normal and area 0; isolated vertices a zero normal.
//  Not on the GPU: the topology edits (pointer-chasing, see HalfEdgeMesh.swift); the directed-edge table build and the
//  Mesh3 weld are in HalfEdgeMesh+GPUTopology.swift.

import Foundation
import simd

public struct HalfEdgeBulk: Sendable {
    public var faceNormals: [Vec3], faceAreas: [Double], vertexNormals: [Vec3]
    public var signedVolume: Double
}

public extension HalfEdgeMesh {
    /// Signed volume over every alive face (fan triangles; positive = wound outward). CPU.
    func signedVolumeAll() -> Double { signedVolume(of: faces) }

    /// Doubles, on the CPU.
    func cpuBulk() -> HalfEdgeBulk {
        let av = faceAreaVectors()
        let vn = vertexNormals()
        return HalfEdgeBulk(faceNormals: av.map { len3($0) > 0 ? $0 / len3($0) : .zero }, faceAreas: av.map { len3($0) / 2 },
                            vertexNormals: vn, signedVolume: signedVolumeAll())
    }

    /// Floats, on the GPU in one command buffer; the CPU twin when there is no device.
    func gpuBulk() -> HalfEdgeBulk { HalfEdgeBulkGPU.run(soa()) ?? cpuBulk() }
}

#if canImport(Metal)
import Metal

enum HalfEdgeBulkGPU {
    static func run(_ s: HalfEdgeSoA) -> HalfEdgeBulk? {
        guard let g = ModellingGPU.shared else { return nil }
        let nf = s.faceHalfedge.count, nv = s.vertexHalfedge.count
        guard nf > 0, nv > 0 else { return nil }
        var centre = SIMD4<Float>.zero
        for p in s.positions { centre += p }
        centre /= Float(Swift.max(s.positions.count, 1))
        let fh = g.buffer(s.faceHalfedge), nx = g.buffer(s.next), pv = g.buffer(s.prev), tw = g.buffer(s.twin)
        let fc = g.buffer(s.face), org = g.buffer(s.origin), vh = g.buffer(s.vertexHalfedge), pos = g.buffer(s.positions)
        let fvec = g.buffer(length: nf * 16), vnrm = g.buffer(length: nv * 16)
        let groups = (nf + 255) / 256
        let partials = g.buffer(length: groups * 4), total = g.buffer(length: 4)
        let ok = g.run { enc in
            g.dispatch1D(enc, "hb_face_vectors", nf, [.buf(fh), .buf(nx), .buf(org), .buf(pos), .buf(fvec), .value(UInt32(nf))])
            g.dispatch1D(enc, "hb_vertex_normals", nv, [.buf(vh), .buf(nx), .buf(pv), .buf(tw), .buf(fc), .buf(fvec), .buf(vnrm), .value(UInt32(nv))])
            g.dispatchGroups(enc, "hb_volume_partials", groups: groups, threads: 256,
                             [.buf(fh), .buf(nx), .buf(org), .buf(pos), .buf(partials), .value(centre), .value(UInt32(nf))])
            g.dispatchGroups(enc, "hb_reduce", groups: 1, threads: 256, [.buf(partials), .buf(total), .value(UInt32(groups))])
        }
        guard ok else { return nil }
        func v3(_ x: SIMD4<Float>) -> Vec3 { Vec3(Double(x.x), Double(x.y), Double(x.z)) }
        let fv = g.read(fvec, SIMD4<Float>.self, count: nf), vn = g.read(vnrm, SIMD4<Float>.self, count: nv)
        return HalfEdgeBulk(
            faceNormals: fv.map { let l = len3(v3($0)); return l > 0 ? v3($0) / l : .zero },
            faceAreas: fv.map { Double($0.w) }, vertexNormals: vn.map(v3),
            signedVolume: Double(total.contents().assumingMemoryBound(to: Float.self)[0]))
    }

    static let source = """
    kernel void hb_face_vectors(device const int* faceHE [[buffer(0)]], device const int* nxt [[buffer(1)]],
                                device const int* org [[buffer(2)]], device const float4* pos [[buffer(3)]],
                                device float4* out [[buffer(4)]], constant uint& n [[buffer(5)]], uint f [[thread_position_in_grid]]) {
        if (f >= n) return;
        int h0 = faceHE[f];
        if (h0 < 0) { out[f] = float4(0); return; }
        float3 sum = 0; int h = h0; int guard = 0;
        do {
            float3 a = pos[org[h]].xyz, b = pos[org[nxt[h]]].xyz;
            sum += float3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
            h = nxt[h]; guard++;
        } while (h != h0 && h >= 0 && guard < 4096);
        out[f] = float4(sum, 0.5f * length(sum));
    }

    kernel void hb_vertex_normals(device const int* vhe [[buffer(0)]], device const int* nxt [[buffer(1)]],
                                  device const int* prv [[buffer(2)]], device const int* twn [[buffer(3)]],
                                  device const int* fac [[buffer(4)]], device const float4* fv [[buffer(5)]],
                                  device float4* out [[buffer(6)]], constant uint& n [[buffer(7)]], uint v [[thread_position_in_grid]]) {
        if (v >= n) return;
        int h0 = vhe[v];
        if (h0 < 0) { out[v] = float4(0); return; }
        float3 sum = fv[fac[h0]].xyz;
        int h = h0; bool closed = false; int guard = 0;
        while (guard++ < 4096) {
            int t = twn[h]; if (t < 0) break;
            h = nxt[t];
            if (h == h0) { closed = true; break; }
            sum += fv[fac[h]].xyz;
        }
        if (!closed) {
            h = h0; guard = 0;
            while (guard++ < 4096) {
                int t = twn[prv[h]]; if (t < 0) break;
                h = t;
                sum += fv[fac[h]].xyz;
            }
        }
        float l = length(sum);
        out[v] = float4(l > 0 ? sum / l : sum, 0);
    }

    // One partial sum per 256-face threadgroup: signed fan volume about `c` (cancels the offset error of a far-off mesh).
    kernel void hb_volume_partials(device const int* faceHE [[buffer(0)]], device const int* nxt [[buffer(1)]],
                                   device const int* org [[buffer(2)]], device const float4* pos [[buffer(3)]],
                                   device float* partials [[buffer(4)]], constant float4& c [[buffer(5)]], constant uint& n [[buffer(6)]],
                                   uint f [[thread_position_in_grid]], uint gid [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup float part[8];
        float vol = 0;
        int h0 = f < n ? faceHE[f] : -1;
        if (h0 >= 0) {
            float3 a = pos[org[h0]].xyz - c.xyz;
            int h = nxt[h0]; int guard = 0;
            while (nxt[h] != h0 && guard++ < 4096) {
                float3 b = pos[org[h]].xyz - c.xyz, d = pos[org[nxt[h]]].xyz - c.xyz;
                vol += dot(a, cross(b, d)) / 6.0f;
                h = nxt[h];
            }
        }
        float s = simd_sum(vol);
        if (lane == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane == 0) { float t = 0; for (uint i = 0; i < 8; i++) t += part[i]; partials[gid] = t; }
    }

    kernel void hb_reduce(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                          uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup float part[8];
        float s = 0;
        for (uint i = tid; i < n; i += 256) s += in[i];
        s = simd_sum(s);
        if (lane == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float t = 0; for (uint i = 0; i < 8; i++) t += part[i]; out[0] = t; }
    }
    """
}
#else
enum HalfEdgeBulkGPU { static func run(_ s: HalfEdgeSoA) -> HalfEdgeBulk? { nil } }
#endif
