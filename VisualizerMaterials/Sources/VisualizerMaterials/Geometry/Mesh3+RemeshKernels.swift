//  Mesh3+RemeshKernels.swift — Metal source of the GPU isotropic remesher: closest point on a BVH of the original
//  surface, the parallel edge split (all long edges at once, with the red/green face patterns), the independent
//  parallel edge flip (valence relaxation, claims by atomic min), and tangential smoothing + projection.
//  Shared pieces (CSR adjacency, feature edges, the collapse step) live in `Mesh3+CollapseKernels*`.
//
//  Parameters (`GP`): split: a = vertices before, b = live faces, c flags (0), w = hi. Smooth: a = V, b = F,
//  x = relaxation. BVH buffers: `blo` / `bhi` node boxes, `bmeta` (first-or-left, count-or-right, isLeaf), and the
//  triangles in leaf order as `ta` / `tb` / `tc`.

import simd

#if canImport(Metal)
extension ModelGPU {
    static let remeshSource = """
    #define BVHA device const float4* blo, device const float4* bhi, device const uint4* bmeta, device const float4* ta, device const float4* tb, device const float4* tc
    #define BVHV blo, bhi, bmeta, ta, tb, tc
    #define BVHK(i) device const float4* blo [[buffer(i)]], device const float4* bhi [[buffer(i+1)]], device const uint4* bmeta [[buffer(i+2)]], device const float4* ta [[buffer(i+3)]], device const float4* tb [[buffer(i+4)]], device const float4* tc [[buffer(i+5)]]

    // Ericson, Real-Time Collision Detection 5.1.5.
    static float3 closest_tri(float3 p, float3 a, float3 b, float3 c) {
        float3 ab = b - a, ac = c - a, ap = p - a;
        float d1 = dot(ab, ap), d2 = dot(ac, ap);
        if (d1 <= 0 && d2 <= 0) return a;
        float3 bp = p - b; float d3 = dot(ab, bp), d4 = dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return b;
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
        float3 cp = p - c; float d5 = dot(ab, cp), d6 = dot(ac, cp);
        if (d6 >= 0 && d5 <= d6) return c;
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        float denom = 1 / (va + vb + vc);
        return a + ab * (vb * denom) + ac * (vc * denom);
    }
    static float box_d2(float3 p, float3 lo, float3 hi) { float3 d = max(max(lo - p, p - hi), float3(0)); return dot(d, d); }

    static float3 bvh_closest(float3 p, BVHA) {
        float best = INFINITY; float3 bp = p;
        uint stack[48]; int sp = 0; stack[sp++] = 0;
        while (sp > 0) {
            uint n = stack[--sp];
            if (box_d2(p, blo[n].xyz, bhi[n].xyz) >= best) continue;
            uint4 m = bmeta[n];
            if (m.z != 0) {
                for (uint i = 0; i < m.y; i++) {
                    uint t = m.x + i;
                    float3 c = closest_tri(p, ta[t].xyz, tb[t].xyz, tc[t].xyz);
                    float3 d = c - p; float d2 = dot(d, d);
                    if (d2 < best) { best = d2; bp = c; }
                }
            } else if (sp < 46) {
                float dl = box_d2(p, blo[m.x].xyz, bhi[m.x].xyz), dr = box_d2(p, blo[m.y].xyz, bhi[m.y].xyz);
                if (dl <= dr) { stack[sp++] = m.y; stack[sp++] = m.x; } else { stack[sp++] = m.x; stack[sp++] = m.y; }
            }
        }
        return bp;
    }

    kernel void bvh_query(device const float4* pts [[buffer(0)]], device float4* out [[buffer(1)]], BVHK(2),
                          constant GP& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
        if (i < p.a) out[i] = float4(bvh_closest(pts[i].xyz, BVHV), 0);
    }
    kernel void fill_u32(device uint* d [[buffer(0)]], constant GP& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
        if (i < p.a) d[i] = p.b;
    }

    // ---- split ----
    static bool edge_long(device const float4* pos, uint a, uint b, float hi) { return length(pos[a].xyz - pos[b].xyz) > hi; }
    // Midpoint id of edge k of face f, or NONE when it is not split. `sc` is the scanned flag array.
    static uint mid_id(MA, device const uint* sc, uint f, uint k, uint nv0, float hi) {
        uint a = tri[3 * f + k], b = tri[3 * f + (k + 1) % 3];
        if (!edge_long(pos, a, b, hi)) return 0xFFFFFFFFu;
        uint s0, s1, cnt = shared_faces(MV, a, b, s0, s1);
        if (cnt != 1 && cnt != 2) return 0xFFFFFFFFu;
        uint o = cnt == 2 ? min(s0, s1) : s0;
        for (uint k2 = 0; k2 < 3; k2++) {
            uint c = tri[3 * o + k2], d = tri[3 * o + (k2 + 1) % 3];
            if ((c == a && d == b) || (c == b && d == a)) return nv0 + sc[3 * o + k2];
        }
        return 0xFFFFFFFFu;
    }
    kernel void rm_split_flag(MA, device uint* flag [[buffer(4)]], constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c > 3 * p.b) return;
        uint s = 0;
        if (c < 3 * p.b) {
            uint f = c / 3, k = c % 3;
            uint a = tri[c], b = tri[3 * f + (k + 1) % 3];
            if (edge_long(pos, a, b, p.w)) {
                uint s0, s1, cnt = shared_faces(MV, a, b, s0, s1);
                if ((cnt == 1 && f == s0) || (cnt == 2 && f == min(s0, s1))) s = 1;
            }
        }
        flag[c] = s;
    }
    // Faces out per face: 1 + number of split edges (0→1, 1→2, 2→3, 3→4).
    kernel void rm_split_count(MA, device const uint* sc [[buffer(4)]], device uint* cnt [[buffer(5)]], constant GP& p [[buffer(15)]],
                               uint f [[thread_position_in_grid]]) {
        if (f > p.b) return;
        uint n = 0;
        if (f < p.b) {
            n = 1;
            for (uint k = 0; k < 3; k++) if (mid_id(MV, sc, f, k, p.a, p.w) != 0xFFFFFFFFu) n++;
        }
        cnt[f] = n;
    }
    // New vertex of each split edge: the midpoint, on the original surface unless the edge is a feature.
    kernel void rm_split_mid(MA, device const uint* sc [[buffer(4)]], BVHK(5), device float4* posW [[buffer(11)]], constant GP& p [[buffer(15)]],
                             uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b || sc[c + 1] == sc[c]) return;
        uint f = c / 3, k = c % 3;
        uint a = tri[c], b = tri[3 * f + (k + 1) % 3];
        uint s0, s1, cnt = shared_faces(MV, a, b, s0, s1);
        float3 m = (pos[a].xyz + pos[b].xyz) * 0.5f;
        if (!edge_feature(MV, (device const float2*)pos, false, p.x, a, b, cnt, s0, s1)) m = bvh_closest(m, BVHV);
        posW[p.a + sc[c]] = float4(m, 0);
    }
    kernel void rm_split_emit(MA, device const uint* sc [[buffer(4)]], device const uint* fo [[buffer(5)]], device uint* out [[buffer(6)]],
                              constant GP& p [[buffer(15)]], uint f [[thread_position_in_grid]]) {
        if (f >= p.b) return;
        uint v[3], m[3], mask = 0;
        for (uint k = 0; k < 3; k++) { v[k] = tri[3 * f + k]; m[k] = mid_id(MV, sc, f, k, p.a, p.w); if (m[k] != 0xFFFFFFFFu) mask |= 1u << k; }
        uint o = 3 * fo[f];
        #define EMIT(A, B, C) { out[o] = A; out[o + 1] = B; out[o + 2] = C; o += 3; }
        uint bits = popcount(mask);
        if (bits == 0) { EMIT(v[0], v[1], v[2]) }
        else if (bits == 1) {
            uint k = mask == 1 ? 0 : (mask == 2 ? 1 : 2);
            uint A = v[k], B = v[(k + 1) % 3], C = v[(k + 2) % 3], M = m[k];
            EMIT(A, M, C) EMIT(M, B, C)
        } else if (bits == 2) {
            uint k = mask == 3 ? 0 : (mask == 6 ? 1 : 2);
            uint A = v[k], B = v[(k + 1) % 3], C = v[(k + 2) % 3], M1 = m[k], M2 = m[(k + 1) % 3];
            EMIT(M1, B, M2)
            if (length(pos[A].xyz - pos[M2].xyz) <= length(pos[M1].xyz - pos[C].xyz)) { EMIT(A, M1, M2) EMIT(A, M2, C) }
            else { EMIT(A, M1, C) EMIT(M1, M2, C) }
        } else {
            EMIT(v[0], m[0], m[2]) EMIT(m[0], v[1], m[1]) EMIT(m[2], m[1], v[2]) EMIT(m[0], m[1], m[2])
        }
    }
    """
}
#endif
