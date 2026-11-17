//  Mesh3+CollapseKernels.swift — the Metal source of the GPU edge-collapse engine shared by decimation and remeshing.
//
//  Data (all device buffers, rebuilt each round where noted): `tri` uint[3F] (original winding), `pos` float4[V],
//  `cuv` float2[3F] (per corner), a vertex → faces CSR (`vs` start / `vl` face ids, rebuilt each round with integer
//  atomics and a prefix scan), per-vertex feature info (`vfeat`: feature-edge count, two feature neighbours, locked),
//  per-vertex quadrics (`q`, three float4 = the CPU's ten numbers). One round: features → a candidate for every edge
//  (all edges in parallel: legality, optimal position, cost) → each vertex's cheapest incident edge → the cheapest
//  over its 1-ring → an edge is chosen when it is cheapest for both ends' rings, which makes chosen edges at least
//  two edges apart so they are independent (their link and flip tests, evaluated on the pre-round mesh, still hold
//  together) → apply (survivor moves, quadrics add, ends are merged) → relabel corners, drop degenerate faces,
//  compact by prefix scan.
//  Same rules as the CPU `TriTopology`: boundary / non-manifold / crease / UV-seam edges are features, a feature
//  vertex only collapses along a feature edge, corners never go, link condition, no folded or squashed triangle.
//  Differences from the CPU: creases are re-detected each round from the current dihedral (the CPU remembers the
//  input's), and collapses go in parallel rounds of locally cheapest edges instead of one global heap.

import simd

#if canImport(Metal)
extension ModelGPU {
    static let collapseSource = """
    #define MAXFACES 40
    #define MAXNB 48
    #define NOCOST 3.0e38f
    #define MA device const uint* tri, device const float4* pos, device const uint* vs, device const uint* vl
    #define MV tri, pos, vs, vl

    struct Cand { float4 p; float cost; uint keep; uint remove; float lerp; uint sh0; uint sh1; uint pad0; uint pad1; };
    struct Quad { float m[10]; };

    static float3 face_raw(device const uint* tri, device const float4* pos, uint f) {
        float3 a = pos[tri[3 * f]].xyz;
        return cross(pos[tri[3 * f + 1]].xyz - a, pos[tri[3 * f + 2]].xyz - a);
    }
    static float3 safe_norm(float3 v) { float l = length(v); return l > 1e-30f ? v / l : float3(0); }
    static bool face_has(device const uint* tri, uint f, uint v) { return tri[3 * f] == v || tri[3 * f + 1] == v || tri[3 * f + 2] == v; }
    static uint corner_of(device const uint* tri, uint f, uint v) { return tri[3 * f] == v ? 0 : (tri[3 * f + 1] == v ? 1 : 2); }
    static uint shared_faces(MA, uint a, uint b, thread uint& s0, thread uint& s1) {
        uint n = 0; s0 = s1 = 0xFFFFFFFFu;
        for (uint i = vs[a]; i < vs[a + 1]; i++) {
            uint f = vl[i];
            if (face_has(tri, f, b)) { if (n == 0) s0 = f; else if (n == 1) s1 = f; n++; }
        }
        return n;
    }
    // Neighbours of v (distinct), up to MAXNB. Returns the count.
    static uint gather_nb(MA, uint v, thread uint* out) {
        uint n = 0;
        for (uint i = vs[v]; i < vs[v + 1]; i++) {
            uint f = vl[i];
            for (uint k = 0; k < 3; k++) {
                uint w = tri[3 * f + k];
                if (w == v) continue;
                bool seen = false;
                for (uint j = 0; j < n; j++) if (out[j] == w) { seen = true; break; }
                if (!seen && n < MAXNB) out[n++] = w;
            }
        }
        return n;
    }
    static bool uv_seam(device const float2* cuv, device const uint* tri, uint a, uint b, uint f1, uint f2) {
        float2 d1 = cuv[3 * f1 + corner_of(tri, f1, a)] - cuv[3 * f2 + corner_of(tri, f2, a)];
        float2 d2 = cuv[3 * f1 + corner_of(tri, f1, b)] - cuv[3 * f2 + corner_of(tri, f2, b)];
        return max(max(fabs(d1.x), fabs(d1.y)), max(fabs(d2.x), fabs(d2.y))) > 1e-6f;
    }
    // Boundary / non-manifold / crease / seam edge (a, b) with `cnt` shared faces s0, s1.
    static bool edge_feature(MA, device const float2* cuv, bool hasUV, float cosC, uint a, uint b, uint cnt, uint s0, uint s1) {
        if (cnt != 2) return true;
        if (dot(safe_norm(face_raw(tri, pos, s0)), safe_norm(face_raw(tri, pos, s1))) < cosC) return true;
        return hasUV && uv_seam(cuv, tri, a, b, s0, s1);
    }

    kernel void zero_u32(device uint* d [[buffer(0)]], constant GP& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
        if (i < p.a) d[i] = 0;
    }
    kernel void vf_count(device const uint* tri [[buffer(0)]], device atomic_uint* cnt [[buffer(1)]], constant GP& p [[buffer(15)]],
                         uint c [[thread_position_in_grid]]) {
        if (c < 3 * p.b) atomic_fetch_add_explicit(&cnt[tri[c]], 1, memory_order_relaxed);
    }
    kernel void vf_cursor(device const uint* vs [[buffer(0)]], device uint* cur [[buffer(1)]], constant GP& p [[buffer(15)]],
                          uint v [[thread_position_in_grid]]) {
        if (v < p.a) cur[v] = vs[v];
    }
    kernel void vf_fill(device const uint* tri [[buffer(0)]], device atomic_uint* cur [[buffer(1)]], device uint* vl [[buffer(2)]],
                        constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b) return;
        uint o = atomic_fetch_add_explicit(&cur[tri[c]], 1, memory_order_relaxed);
        vl[o] = c / 3;
    }

    // vfeat[v] = (feature-edge count, first, second feature neighbour, locked). p.a = V, c bit0 hasUV, x = cos crease.
    kernel void vtx_feat(MA, device const float2* cuv [[buffer(4)]], device uint4* vfeat [[buffer(5)]],
                         constant GP& p [[buffer(15)]], uint v [[thread_position_in_grid]]) {
        if (v >= p.a) return;
        uint nf = vs[v + 1] - vs[v];
        uint4 out = uint4(0, 0, 0, (nf == 0 || nf > MAXFACES) ? 1 : 0);
        if (out.w == 0) {
            for (uint i = vs[v]; i < vs[v + 1]; i++) {
                uint f = vl[i];
                for (uint k = 0; k < 3; k++) {
                    uint w = tri[3 * f + k];
                    if (w == v) continue;
                    bool first = true;
                    for (uint j = vs[v]; j < i; j++) if (face_has(tri, vl[j], w)) { first = false; break; }
                    if (!first) continue;
                    uint s0, s1, cnt = shared_faces(MV, v, w, s0, s1);
                    if (edge_feature(MV, cuv, (p.c & 1) != 0, p.x, v, w, cnt, s0, s1)) {
                        if (out.x == 0) out.y = w; else if (out.x == 1) out.z = w;
                        out.x++;
                    }
                }
            }
        }
        vfeat[v] = out;
    }

    // ---- quadrics ----
    static Quad qplane(float3 n, float d, float w) {
        Quad q;
        q.m[0] = n.x * n.x * w; q.m[1] = n.x * n.y * w; q.m[2] = n.x * n.z * w; q.m[3] = n.x * d * w;
        q.m[4] = n.y * n.y * w; q.m[5] = n.y * n.z * w; q.m[6] = n.y * d * w; q.m[7] = n.z * n.z * w;
        q.m[8] = n.z * d * w; q.m[9] = d * d * w;
        return q;
    }
    static Quad qload(device const float4* q, uint v) {
        Quad o; float4 a = q[3 * v], b = q[3 * v + 1], c = q[3 * v + 2];
        o.m[0] = a.x; o.m[1] = a.y; o.m[2] = a.z; o.m[3] = a.w; o.m[4] = b.x; o.m[5] = b.y; o.m[6] = b.z; o.m[7] = b.w;
        o.m[8] = c.x; o.m[9] = c.y;
        return o;
    }
    static void qstore(device float4* q, uint v, thread const Quad& o) {
        q[3 * v] = float4(o.m[0], o.m[1], o.m[2], o.m[3]); q[3 * v + 1] = float4(o.m[4], o.m[5], o.m[6], o.m[7]);
        q[3 * v + 2] = float4(o.m[8], o.m[9], 0, 0);
    }
    static float qerror(thread const Quad& q, float3 p) {
        float x = p.x, y = p.y, z = p.z;
        float v = q.m[0] * x * x + 2 * q.m[1] * x * y + 2 * q.m[2] * x * z + 2 * q.m[3] * x
                + q.m[4] * y * y + 2 * q.m[5] * y * z + 2 * q.m[6] * y + q.m[7] * z * z + 2 * q.m[8] * z + q.m[9];
        return max(v, 0.0f);
    }
    static bool qopt(thread const Quad& q, thread float3& out) {
        float a = q.m[0], b = q.m[1], c = q.m[2], e = q.m[4], f = q.m[5], h = q.m[7];
        float det = a * (e * h - f * f) - b * (b * h - f * c) + c * (b * f - e * c);
        float sc = max(max(fabs(a), fabs(e)), max(fabs(h), 1e-30f));
        if (fabs(det) <= 1e-6f * sc * sc * sc) return false;
        float3 r = float3(-q.m[3], -q.m[6], -q.m[8]);
        out = float3((r.x * (e * h - f * f) - b * (r.y * h - f * r.z) + c * (r.y * f - e * r.z)) / det,
                     (a * (r.y * h - f * r.z) - r.x * (b * h - f * c) + c * (b * r.z - r.y * c)) / det,
                     (a * (e * r.z - r.y * f) - b * (b * r.z - r.y * c) + r.x * (b * f - e * c)) / det);
        return true;
    }

    // The same quadric seen from a frame whose origin is `s` further along (error(y) with y' = y + s).
    static Quad qshift(thread const Quad& q, float3 s) {
        Quad o = q;
        float3 As = float3(q.m[0] * s.x + q.m[1] * s.y + q.m[2] * s.z, q.m[1] * s.x + q.m[4] * s.y + q.m[5] * s.z,
                           q.m[2] * s.x + q.m[5] * s.y + q.m[7] * s.z);
        o.m[3] = q.m[3] + As.x; o.m[6] = q.m[6] + As.y; o.m[8] = q.m[8] + As.z;
        o.m[9] = q.m[9] + 2 * (q.m[3] * s.x + q.m[6] * s.y + q.m[8] * s.z) + dot(As, s);
        return o;
    }

    // Per-vertex: area-weighted face planes plus constraint planes along boundary / non-manifold / crease edges.
    kernel void quad_init(MA, device float4* q [[buffer(4)]], constant GP& p [[buffer(15)]], uint v [[thread_position_in_grid]]) {
        if (v >= p.a) return;
        Quad acc; for (uint i = 0; i < 10; i++) acc.m[i] = 0;
        float3 av = pos[v].xyz;       // the vertex's own frame: quadrics are kept about it so Float keeps its digits
        for (uint i = vs[v]; i < vs[v + 1]; i++) {
            uint f = vl[i];
            float3 n = face_raw(tri, pos, f);
            float a2 = length(n);
            if (a2 > 1e-14f) {
                float3 u = n / a2;
                Quad qf = qplane(u, -dot(u, pos[tri[3 * f]].xyz - av), a2 * 0.5f);
                for (uint j = 0; j < 10; j++) acc.m[j] += qf.m[j];
            }
            uint k = corner_of(tri, f, v);
            for (uint s = 0; s < 2; s++) {
                uint a = s == 0 ? v : tri[3 * f + (k + 2) % 3], b = s == 0 ? tri[3 * f + (k + 1) % 3] : v;
                uint other = s == 0 ? b : a;
                uint s0, s1, cnt = shared_faces(MV, v, other, s0, s1);
                bool crease = cnt == 2 && dot(safe_norm(face_raw(tri, pos, s0)), safe_norm(face_raw(tri, pos, s1))) < p.x;
                if (!(cnt == 1 || cnt > 2 || crease)) continue;
                float3 e = pos[b].xyz - pos[a].xyz;
                float len = length(e);
                float3 m = cross(e, safe_norm(n));
                if (len > 1e-9f && length(m) > 1e-9f) {
                    float3 mn = normalize(m);
                    Quad qc = qplane(mn, -dot(mn, pos[a].xyz - av), len * len);
                    for (uint j = 0; j < 10; j++) acc.m[j] += qc.m[j];
                }
            }
        }
        qstore(q, v, acc);
    }
    """
}
#endif
