//  Mesh3+RemeshKernels2.swift — the second half of the GPU remesher's Metal source: valence, independent parallel
//  edge flips (an edge flips when it improves the valence deviation of its four corners and wins its two faces and
//  its opposite-vertex pair by atomic min, so flips never touch the same face or create the same new edge twice),
//  and tangential smoothing projected back onto the original surface through the BVH.
//  Same rules as `TriTopology.flip`: interior non-feature edges only, no edge between corners already joined, no
//  folded or squashed result.

import simd

#if canImport(Metal)
extension ModelGPU {
    static let remeshSource2 = """
    // val[v] = (distinct neighbours, incident faces).
    kernel void rm_val(MA, device uint2* val [[buffer(4)]], constant GP& p [[buffer(15)]], uint v [[thread_position_in_grid]]) {
        if (v >= p.a) return;
        uint nb[MAXNB];
        val[v] = uint2(gather_nb(MV, v, nb), vs[v + 1] - vs[v]);
    }
    static uint third_of(device const uint* tri, uint f, uint a, uint b) {
        for (uint k = 0; k < 3; k++) { uint w = tri[3 * f + k]; if (w != a && w != b) return w; }
        return a;
    }
    static int vtarget(uint2 v) { return v.y < v.x ? 4 : 6; }

    kernel void rm_flip_eval(MA, device const uint4* vfeat [[buffer(4)]], device const uint2* val [[buffer(5)]],
                             device uint4* fc [[buffer(6)]], constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b) return;
        uint4 out = uint4(0);
        uint f = c / 3, k = c % 3;
        uint a = tri[c], b = tri[3 * f + (k + 1) % 3];
        uint s0, s1, cnt = shared_faces(MV, a, b, s0, s1);
        if (cnt == 2 && f == min(s0, s1) && !edge_feature(MV, (device const float2*)pos, false, p.x, a, b, cnt, s0, s1)) {
            uint f2 = f == s0 ? s1 : s0;
            uint x1 = third_of(tri, f, a, b), x2 = third_of(tri, f2, a, b);
            bool ok = x1 != x2 && vfeat[a].w == 0 && vfeat[b].w == 0 && vfeat[x1].w == 0 && vfeat[x2].w == 0;
            for (uint i = vs[x1]; ok && i < vs[x1 + 1]; i++) if (face_has(tri, vl[i], x2)) ok = false;
            if (ok) {
                float3 pa = pos[a].xyz, pb = pos[b].xyz, p1 = pos[x1].xyz, p2 = pos[x2].xyz;
                float3 n1 = cross(p2 - pa, p1 - pa), n2 = cross(pb - p2, p1 - p2);
                float3 o = face_raw(tri, pos, f) + face_raw(tri, pos, f2);
                float l1 = length(n1), l2 = length(n2);
                ok = dot(n1, o) > 0 && dot(n2, o) > 0 && l1 > 1e-14f && l2 > 1e-14f && dot(n1, n2) > 0.5f * l1 * l2;
            }
            if (ok) {
                uint vv[4] = { a, b, x1, x2 };
                int before = 0, after = 0;
                for (uint i = 0; i < 4; i++) {
                    int d = (int)val[vv[i]].x, tg = vtarget(val[vv[i]]);
                    int nd = i < 2 ? d - 1 : d + 1;
                    before += (d - tg) * (d - tg); after += (nd - tg) * (nd - tg);
                }
                if (after < before) out = uint4(f2, x1, x2, 1);
            }
        }
        fc[c] = out;
    }
    kernel void rm_flip_claim(device const uint4* fc [[buffer(0)]], device atomic_uint* claimF [[buffer(1)]],
                              device atomic_uint* claimV [[buffer(2)]], constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b || fc[c].w == 0) return;
        atomic_fetch_min_explicit(&claimF[c / 3], c, memory_order_relaxed);
        atomic_fetch_min_explicit(&claimF[fc[c].x], c, memory_order_relaxed);
        atomic_fetch_min_explicit(&claimV[min(fc[c].y, fc[c].z)], c, memory_order_relaxed);
    }
    kernel void rm_flip_apply(device uint* tri [[buffer(0)]], device const uint4* fc [[buffer(1)]], device const uint* claimF [[buffer(2)]],
                              device const uint* claimV [[buffer(3)]], device atomic_uint* done [[buffer(4)]], constant GP& p [[buffer(15)]],
                              uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b || fc[c].w == 0) return;
        uint f = c / 3, k = c % 3, f2 = fc[c].x, x1 = fc[c].y, x2 = fc[c].z;
        if (claimF[f] != c || claimF[f2] != c || claimV[min(x1, x2)] != c) return;
        uint a = tri[c], b = tri[3 * f + (k + 1) % 3];
        tri[3 * f] = a; tri[3 * f + 1] = x2; tri[3 * f + 2] = x1;
        tri[3 * f2] = x2; tri[3 * f2 + 1] = b; tri[3 * f2 + 2] = x1;
        atomic_fetch_add_explicit(done, 1, memory_order_relaxed);
    }

    // Move each non-feature vertex toward its neighbours' centroid within the tangent plane, then onto the surface.
    kernel void rm_smooth(MA, device const uint4* vfeat [[buffer(4)]], BVHK(5), device float4* posOut [[buffer(11)]],
                          constant GP& p [[buffer(15)]], uint v [[thread_position_in_grid]]) {
        if (v >= p.a) return;
        posOut[v] = pos[v];
        uint4 ft = vfeat[v];
        if (vs[v + 1] == vs[v] || ft.w != 0 || ft.x != 0) return;
        uint nb[MAXNB];
        uint n = gather_nb(MV, v, nb);
        if (n == 0) return;
        float3 cen = float3(0);
        for (uint i = 0; i < n; i++) cen += pos[nb[i]].xyz;
        cen /= (float)n;
        float3 nrm = float3(0);
        for (uint i = vs[v]; i < vs[v + 1]; i++) nrm += face_raw(tri, pos, vl[i]);
        nrm = safe_norm(nrm);
        float3 pv = pos[v].xyz, d = cen - pv;
        d -= nrm * dot(d, nrm);
        posOut[v] = float4(bvh_closest(pv + d * p.x, BVHV), 0);
    }
    """
}
#endif
