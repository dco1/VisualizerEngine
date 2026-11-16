//  Modifier+GPUBVH.swift — the Shrinkwrap target as flat buffers, and the Metal traversal (ticket item M2).
//
//  `TriangleBVH` keeps its nodes private, so this builds its own tree the same way (top-down median split on the
//  longest centroid axis, leaves of ≤ 4 triangles) on the CPU, then lays it out flat for the GPU: 48-byte nodes
//  (box padded one Float ulp outward so rounding can never cull a true hit), triangles re-ordered into leaf order,
//  and per triangle three vertex normals + the face normal (so the kernel's "interpolated normal, else face normal"
//  needs no branches on the mesh). The build itself stays on the CPU (it is a sort per level; an LBVH build on the
//  GPU is the follow-up); the queries, which are 200k–millions per modifier, are the GPU's.
//  Guarantees: closest point / nearest ray hit equal `TriangleBVH` within Float rounding. The ray test accepts
//  within 1e-6 of a triangle's edge (the CPU's 1e-9 is below Float resolution) and ignores hits closer than 1e-7.

#if canImport(Metal)
import Metal
import simd

struct GPUBVHNode { var lo: SIMD4<Float>; var hi: SIMD4<Float>; var left: Int32; var right: Int32; var start: Int32; var count: Int32 }

struct FlatBVH {
    var nodes: [GPUBVHNode] = []
    var a: [SIMD4<Float>] = [], b: [SIMD4<Float>] = [], c: [SIMD4<Float>] = []
    var n: [SIMD4<Float>] = []                     // 4 per triangle: na, nb, nc, face normal

    init(_ m: Mesh3) {
        var A: [Vec3] = [], B: [Vec3] = [], C: [Vec3] = [], tri: [Int] = []
        var t = 0
        while t + 2 < m.indices.count {
            A.append(m.positions[Int(m.indices[t])]); B.append(m.positions[Int(m.indices[t + 1])]); C.append(m.positions[Int(m.indices[t + 2])])
            tri.append(t / 3); t += 3
        }
        guard !A.isEmpty else { return }
        var order = Array(0..<A.count)
        let cen = (0..<A.count).map { (A[$0] + B[$0] + C[$0]) / 3 }
        func down(_ x: Double) -> Float { Float(x).nextDown }
        func up(_ x: Double) -> Float { Float(x).nextUp }
        func build(_ s: Int, _ e: Int) -> Int32 {
            var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity), clo = lo, chi = hi
            for k in s..<e {
                let i = order[k]
                lo = simd.min(lo, simd.min(A[i], simd.min(B[i], C[i]))); hi = simd.max(hi, simd.max(A[i], simd.max(B[i], C[i])))
                clo = simd.min(clo, cen[i]); chi = simd.max(chi, cen[i])
            }
            let idx = Int32(nodes.count)
            nodes.append(GPUBVHNode(lo: SIMD4(down(lo.x), down(lo.y), down(lo.z), 0), hi: SIMD4(up(hi.x), up(hi.y), up(hi.z), 0),
                                    left: -1, right: -1, start: Int32(s), count: Int32(e - s)))
            if e - s <= 4 { return idx }
            let ext = chi - clo
            let axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : (ext.y >= ext.z ? 1 : 2)
            order[s..<e].sort { cen[$0][axis] < cen[$1][axis] }
            let mid = (s + e) / 2
            let l = build(s, mid), r = build(mid, e)
            nodes[Int(idx)].left = l; nodes[Int(idx)].right = r; nodes[Int(idx)].count = 0
            return idx
        }
        _ = build(0, A.count)
        func f4(_ v: Vec3) -> SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), 0) }
        let hasN = m.normals.count == m.positions.count
        for i in order {
            a.append(f4(A[i])); b.append(f4(B[i])); c.append(f4(C[i]))
            let fnv = cross3(B[i] - A[i], C[i] - A[i])
            let face = len3(fnv) > 1e-14 ? normalize3(fnv) : Vec3(0, 1, 0)
            if hasN {
                let k = tri[i] * 3
                n.append(f4(m.normals[Int(m.indices[k])])); n.append(f4(m.normals[Int(m.indices[k + 1])])); n.append(f4(m.normals[Int(m.indices[k + 2])]))
            } else { n.append(.zero); n.append(.zero); n.append(.zero) }
            n.append(f4(face))
        }
    }
}

extension ModifierKernels {
    static let shrinkwrap = """
    struct BNode { float4 lo; float4 hi; int left; int right; int start; int count; };

    static float3 cp_tri(float3 p, float3 a, float3 b, float3 c) {
        float3 ab = b - a, ac = c - a, ap = p - a;
        float d1 = dot(ab, ap), d2 = dot(ac, ap);
        if (d1 <= 0 && d2 <= 0) return a;
        float3 bp = p - b; float d3 = dot(ab, bp), d4 = dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return b;
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
        float3 cpv = p - c; float d5 = dot(ab, cpv), d6 = dot(ac, cpv);
        if (d6 >= 0 && d5 <= d6) return c;
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        float den = 1 / (va + vb + vc);
        return a + ab * (vb * den) + ac * (vc * den);
    }
    static float3 bary_tri(float3 p, float3 a, float3 b, float3 c) {
        float3 v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1), d20 = dot(v2, v0), d21 = dot(v2, v1);
        float den = d00 * d11 - d01 * d01;
        if (fabs(den) <= 1e-30f) return float3(1, 0, 0);
        float v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den;
        return float3(1 - v - w, v, w);
    }
    static bool bvh_closest(float3 p, device const BNode* nodes, device const float4* A, device const float4* B,
                            device const float4* C, thread float3& q, thread int& tri, thread float3& bary) {
        int stack[64]; int sp = 0; stack[sp++] = 0;
        float bestD2 = INFINITY; bool found = false;
        while (sp > 0) {
            BNode nd = nodes[stack[--sp]];
            float3 d = max(max(nd.lo.xyz - p, p - nd.hi.xyz), float3(0));
            if (dot(d, d) > bestD2) continue;
            if (nd.left < 0) {
                for (int i = nd.start; i < nd.start + nd.count; i++) {
                    float3 qq = cp_tri(p, A[i].xyz, B[i].xyz, C[i].xyz);
                    float d2 = dot(qq - p, qq - p);
                    if (d2 <= bestD2) { bestD2 = d2; q = qq; tri = i; bary = bary_tri(qq, A[i].xyz, B[i].xyz, C[i].xyz); found = true; }
                }
            } else {
                BNode l = nodes[nd.left], r = nodes[nd.right];
                float3 cl = (l.lo.xyz + l.hi.xyz) / 2 - p, cr = (r.lo.xyz + r.hi.xyz) / 2 - p;
                if (dot(cl, cl) < dot(cr, cr)) { stack[sp++] = nd.right; stack[sp++] = nd.left; }
                else { stack[sp++] = nd.left; stack[sp++] = nd.right; }
            }
        }
        return found;
    }
    static bool bvh_ray(float3 o, float3 d, device const BNode* nodes, device const float4* A, device const float4* B,
                        device const float4* C, thread float& bestT, thread int& tri, thread float3& bary) {
        int stack[64]; int sp = 0; stack[sp++] = 0;
        bool found = false;
        float3 inv = float3(1 / (d.x == 0 ? 1e-30f : d.x), 1 / (d.y == 0 ? 1e-30f : d.y), 1 / (d.z == 0 ? 1e-30f : d.z));
        while (sp > 0) {
            BNode nd = nodes[stack[--sp]];
            float3 t1 = (nd.lo.xyz - 1e-6f - o) * inv, t2 = (nd.hi.xyz + 1e-6f - o) * inv;
            float3 tmin = min(t1, t2), tmax = max(t1, t2);
            float enter = max(tmin.x, max(tmin.y, tmin.z)), ex = min(tmax.x, min(tmax.y, tmax.z));
            if (ex < max(enter, 0.0f) || enter > bestT) continue;
            if (nd.left < 0) {
                for (int i = nd.start; i < nd.start + nd.count; i++) {
                    float3 a = A[i].xyz, e1 = B[i].xyz - a, e2 = C[i].xyz - a, pv = cross(d, e2);
                    float det = dot(e1, pv);
                    if (fabs(det) <= 1e-14f) continue;
                    float idet = 1 / det; float3 tv = o - a;
                    float u = dot(tv, pv) * idet;
                    if (u < -1e-6f || u > 1 + 1e-6f) continue;
                    float3 qv = cross(tv, e1); float v = dot(d, qv) * idet;
                    if (v < -1e-6f || u + v > 1 + 1e-6f) continue;
                    float t = dot(e2, qv) * idet;
                    if (t > 1e-7f && t < bestT) { bestT = t; tri = i; bary = float3(1 - u - v, u, v); found = true; }
                }
            } else { stack[sp++] = nd.left; stack[sp++] = nd.right; }
        }
        return found;
    }
    static float3 tri_normal(device const float4* N, int tri, float3 w) {
        float3 n = N[4 * tri].xyz * w.x + N[4 * tri + 1].xyz * w.y + N[4 * tri + 2].xyz * w.z;
        return length(n) > 1e-9f ? normalize(n) : N[4 * tri + 3].xyz;
    }

    struct SWP { float4 fixedDir; uint n, mode, tryPos, tryNeg; float offset, factor, pad0, pad1; };
    kernel void shrinkwrap(device const float4* pos [[buffer(0)]], device float4* outP [[buffer(1)]],
                           device const uint* cid [[buffer(2)]], device const uint* rep [[buffer(3)]],
                           device const float4* wN [[buffer(4)]], device const BNode* nodes [[buffer(5)]],
                           device const float4* A [[buffer(6)]], device const float4* B [[buffer(7)]],
                           device const float4* C [[buffer(8)]], device const float4* N [[buffer(9)]],
                           constant SWP& P [[buffer(10)]], uint v [[thread_position_in_grid]]) {
        if (v >= P.n) return;
        uint id = cid[v];
        float3 p = pos[rep[id]].xyz, cur = pos[v].xyz;
        float3 q, nrm; bool hit = false; int tri = 0; float3 bary;
        if (P.mode == 0) {
            hit = bvh_closest(p, nodes, A, B, C, q, tri, bary);
            if (hit) nrm = tri_normal(N, tri, bary);
        } else {
            float3 dir = P.mode == 1 ? wN[id].xyz : normalize(P.fixedDir.xyz);
            float bestT = INFINITY; int bt = 0; float3 bb = float3(0);
            if (P.tryPos != 0 && bvh_ray(p, dir, nodes, A, B, C, bestT, bt, bb)) { hit = true; tri = bt; bary = bb; q = p + dir * bestT; }
            // the reverse ray only counts when strictly nearer than the forward hit (the CPU's h.t < best.t)
            if (P.tryNeg != 0 && bvh_ray(p, -dir, nodes, A, B, C, bestT, bt, bb)) { hit = true; tri = bt; bary = bb; q = p - dir * bestT; }
            if (hit) nrm = tri_normal(N, tri, bary);
        }
        outP[v] = hit ? float4(cur + (q + nrm * P.offset - cur) * P.factor, 0) : float4(cur, 0);
    }
    """
}
#endif
