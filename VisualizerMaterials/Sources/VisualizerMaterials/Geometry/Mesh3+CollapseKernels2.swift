//  Mesh3+CollapseKernels2.swift — second half of the collapse engine's Metal source: the all-edges candidate
//  evaluation (legality, optimal position, cost), the independent-set selection, and applying a round of collapses.
//  See `Mesh3+CollapseKernels.swift` for the data layout and the round's structure.
//
//  Parameters (`GP`): a = vertex count, b = live face count, c = flags (bit0 UVs, bit1 remesh mode), d = live face
//  count (for the "more than 4 left" guard), x = cos(crease angle), y = cos(corner turn), z = remesh `lo`, w = `hi`.

import simd

#if canImport(Metal)
extension ModelGPU {
    static let collapseSource2 = """
    static bool is_corner(device const float4* pos, uint w, uint4 ff, float cosTurn) {
        if (ff.x != 2) return true;
        float3 d1 = safe_norm(pos[w].xyz - pos[ff.y].xyz), d2 = safe_norm(pos[ff.z].xyz - pos[w].xyz);
        return dot(d1, d2) < cosTurn;
    }
    // Would moving vertex v of face f to P fold or squash it?
    static bool flips(MA, uint f, uint v, float3 P) {
        float3 n0 = face_raw(tri, pos, f);
        float l0 = length(n0);
        if (l0 < 1e-14f) return false;
        float3 q[3];
        for (uint k = 0; k < 3; k++) { uint w = tri[3 * f + k]; q[k] = w == v ? P : pos[w].xyz; }
        float3 n1 = cross(q[1] - q[0], q[2] - q[0]);
        float l1 = length(n1);
        return l1 < l0 * 1e-3f || dot(n0, n1) < 0.25f * l0 * l1;
    }

    kernel void cand_eval(MA, device const float2* cuv [[buffer(4)]], device const uint4* vfeat [[buffer(5)]],
                          device const float4* q [[buffer(6)]], device Cand* cand [[buffer(7)]],
                          device const float4* anchor [[buffer(8)]],
                          constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b) return;
        Cand out;
        out.p = float4(0); out.cost = NOCOST; out.keep = 0; out.remove = 0; out.lerp = -1;
        out.sh0 = out.sh1 = 0xFFFFFFFFu; out.pad0 = out.pad1 = 0;
        uint f = c / 3, k = c % 3;
        uint a = tri[c], b = tri[3 * f + (k + 1) % 3];
        uint u = min(a, b), v = max(a, b);
        uint s0, s1, cnt = shared_faces(MV, u, v, s0, s1);
        bool owner = (cnt == 1 && f == s0) || (cnt == 2 && f == min(s0, s1));
        uint4 fu = vfeat[u], fv = vfeat[v];
        bool remesh = (p.c & 2) != 0, hasUV = (p.c & 1) != 0;
        float3 pu = pos[u].xyz, pv = pos[v].xyz;
        float len = length(pv - pu);
        if (owner && fu.w == 0 && fv.w == 0 && p.d > 4 && (!remesh || len < p.z)) {
            bool free = fu.x == 0 && fv.x == 0;
            bool edgeFeat = free ? false : edge_feature(MV, cuv, hasUV, p.x, u, v, cnt, s0, s1);
            bool remU = free || fu.x == 0 || (edgeFeat && !is_corner(pos, u, fu, p.y));
            bool remV = free || fv.x == 0 || (edgeFeat && !is_corner(pos, v, fv, p.y));
            if (remU || remV) {
                float3 cp[4]; uint ck[4], cr[4]; float cl[4], cc[4]; int nc = 0;
                Quad qe;
                float3 ou = anchor[u].xyz;     // quadrics are about each vertex's anchor; evaluate in u's frame
                if (!remesh) { Quad qa = qload(q, u), qb = qload(q, v); qb = qshift(qb, ou - anchor[v].xyz); for (uint i = 0; i < 10; i++) qe.m[i] = qa.m[i] + qb.m[i]; }
                float3 mid = (pu + pv) * 0.5f, ab = pv - pu;
                if (free) {
                    float3 cands[4]; int n = 0;
                    cands[n++] = mid;
                    if (!remesh) {
                        cands[n++] = pu; cands[n++] = pv;
                        float3 o;
                        if (qopt(qe, o) && length(o + ou - mid) < 2 * length(ab)) cands[n++] = o + ou;
                    }
                    for (int i = 0; i < n; i++) {
                        float t = clamp(dot(cands[i] - pu, ab) / max(dot(ab, ab), 1e-30f), 0.0f, 1.0f);
                        cp[nc] = cands[i]; ck[nc] = v; cr[nc] = u; cl[nc] = remesh ? 0.5f : t;
                        cc[nc] = remesh ? len : qerror(qe, cands[i] - ou); nc++;
                    }
                } else {
                    if (remU) { cp[nc] = pv; ck[nc] = v; cr[nc] = u; cl[nc] = -1; cc[nc] = remesh ? len : qerror(qe, pv - ou); nc++; }
                    if (remV) { cp[nc] = pu; ck[nc] = u; cr[nc] = v; cl[nc] = -1; cc[nc] = remesh ? len : qerror(qe, pu - ou); nc++; }
                }
                if (!remesh) {      // cheapest first (insertion sort)
                    for (int i = 1; i < nc; i++) {
                        float3 tp = cp[i]; uint tk = ck[i], tr = cr[i]; float tl = cl[i], tc = cc[i];
                        int j = i - 1;
                        while (j >= 0 && cc[j] > tc) { cp[j + 1] = cp[j]; ck[j + 1] = ck[j]; cr[j + 1] = cr[j]; cl[j + 1] = cl[j]; cc[j + 1] = cc[j]; j--; }
                        cp[j + 1] = tp; ck[j + 1] = tk; cr[j + 1] = tr; cl[j + 1] = tl; cc[j + 1] = tc;
                    }
                }
                uint nbu[MAXNB], nbv[MAXNB];
                uint nu = gather_nb(MV, u, nbu), nv = gather_nb(MV, v, nbv);
                uint common = 0;
                for (uint i = 0; i < nv; i++) for (uint j = 0; j < nu; j++) if (nbv[i] == nbu[j]) { common++; break; }
                if (common == cnt) {
                    for (int j = 0; j < nc; j++) {
                        uint kp = ck[j], rm = cr[j]; float3 P = cp[j];
                        bool ok = true;
                        if (remesh) {
                            for (uint i = 0; i < nu && ok; i++) if (nbu[i] != v && length(P - pos[nbu[i]].xyz) > p.w) ok = false;
                            for (uint i = 0; i < nv && ok; i++) if (nbv[i] != u && length(P - pos[nbv[i]].xyz) > p.w) ok = false;
                        }
                        for (uint i = vs[rm]; i < vs[rm + 1] && ok; i++) {
                            uint g = vl[i];
                            if (g != s0 && g != s1 && flips(MV, g, rm, P)) ok = false;
                        }
                        if (ok && any(P != pos[kp].xyz)) {
                            for (uint i = vs[kp]; i < vs[kp + 1] && ok; i++) {
                                uint g = vl[i];
                                if (g != s0 && g != s1 && flips(MV, g, kp, P)) ok = false;
                            }
                        }
                        if (ok) {
                            out.p = float4(P, 0); out.cost = cc[j]; out.keep = kp; out.remove = rm; out.lerp = cl[j];
                            out.sh0 = s0; out.sh1 = cnt == 2 ? s1 : 0xFFFFFFFFu;
                            break;
                        }
                    }
                }
            }
        }
        cand[c] = out;
    }

    static ulong cand_key(float cost, uint c) { return ((ulong)as_type<uint>(cost) << 32) | (ulong)c; }

    // The cheapest candidate edge touching each vertex.
    kernel void edge_min1(MA, device const Cand* cand [[buffer(4)]], device ulong* m1 [[buffer(5)]],
                          constant GP& p [[buffer(15)]], uint v [[thread_position_in_grid]]) {
        if (v >= p.a) return;
        ulong best = 0xFFFFFFFFFFFFFFFFul;
        for (uint i = vs[v]; i < vs[v + 1]; i++) {
            uint f = vl[i];
            for (uint k = 0; k < 3; k++) {
                uint c = 3 * f + k;
                if (tri[c] != v && tri[3 * f + (k + 1) % 3] != v) continue;
                float cost = cand[c].cost;
                if (cost < NOCOST) best = min(best, cand_key(cost, c));
            }
        }
        m1[v] = best;
    }
    // The cheapest candidate edge touching v or any neighbour of v.
    kernel void edge_min2(MA, device const ulong* m1 [[buffer(4)]], device ulong* m2 [[buffer(5)]],
                          constant GP& p [[buffer(15)]], uint v [[thread_position_in_grid]]) {
        if (v >= p.a) return;
        ulong best = m1[v];
        for (uint i = vs[v]; i < vs[v + 1]; i++) {
            uint f = vl[i];
            for (uint k = 0; k < 3; k++) best = min(best, m1[tri[3 * f + k]]);
        }
        m2[v] = best;
    }
    // A candidate is taken when it is the cheapest within both ends' rings (and under the budget threshold c:d).
    kernel void edge_select(device const Cand* cand [[buffer(0)]], device const ulong* m2 [[buffer(1)]], device uint* sel [[buffer(2)]],
                            constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c > 3 * p.b) return;
        uint s = 0;
        if (c < 3 * p.b && cand[c].cost < NOCOST) {
            ulong key = cand_key(cand[c].cost, c), thr = ((ulong)p.c << 32) | (ulong)p.d;
            s = (key == m2[cand[c].keep] && key == m2[cand[c].remove] && key <= thr) ? 1 : 0;
        }
        sel[c] = s;
    }

    kernel void vmap_init(device uint* vmap [[buffer(0)]], device float4* vuv [[buffer(1)]], constant GP& p [[buffer(15)]],
                          uint v [[thread_position_in_grid]]) {
        if (v >= p.a) return;
        vmap[v] = v; vuv[v] = float4(0);
    }
    // `sc` is the scanned selection (selected iff sc[c + 1] != sc[c]).
    kernel void collapse_apply(device float4* pos [[buffer(0)]], device float4* q [[buffer(1)]], device const Cand* cand [[buffer(2)]],
                               device const uint* sc [[buffer(3)]], device uint* vmap [[buffer(4)]], device uint2* rec [[buffer(5)]],
                               device float4* vuv [[buffer(6)]], device const uint* tri [[buffer(7)]], device const float2* cuv [[buffer(8)]],
                               device uint* moved [[buffer(9)]], device const float4* anchor [[buffer(10)]], constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b || sc[c + 1] == sc[c]) return;
        Cand cd = cand[c];
        uint kp = cd.keep, rm = cd.remove;
        if (any(cd.p.xyz != pos[kp].xyz)) moved[kp] = 1;
        pos[kp] = float4(cd.p.xyz, 0);
        Quad a = qload(q, kp), b = qload(q, rm);
        b = qshift(b, anchor[kp].xyz - anchor[rm].xyz);
        for (uint i = 0; i < 10; i++) a.m[i] += b.m[i];
        qstore(q, kp, a);
        vmap[rm] = kp;
        rec[rm] = uint2(cd.sh0, cd.sh1);
        if ((p.c & 1) != 0 && cd.lerp >= 0) {
            uint s = cd.sh0;
            float2 uvn = cuv[3 * s + corner_of(tri, s, rm)] * (1 - cd.lerp) + cuv[3 * s + corner_of(tri, s, kp)] * cd.lerp;
            vuv[rm] = float4(uvn, 1, 0); vuv[kp] = float4(uvn, 1, 0);
        }
    }
    kernel void relabel(device const uint* tri [[buffer(0)]], device const float2* cuv [[buffer(1)]], device const uint* vmap [[buffer(2)]],
                        device const uint2* rec [[buffer(3)]], device const float4* vuv [[buffer(4)]], device uint* triOut [[buffer(5)]],
                        device float2* cuvOut [[buffer(6)]], constant GP& p [[buffer(15)]], uint c [[thread_position_in_grid]]) {
        if (c >= 3 * p.b) return;
        uint v = tri[c], nv = vmap[v];
        triOut[c] = nv;
        if ((p.c & 1) == 0) return;
        float2 uvc = cuv[c];
        float4 w = vuv[v];
        if (w.z > 0.5f) uvc = w.xy;
        else if (nv != v) {
            uint2 r = rec[v];
            for (uint j = 0; j < 2; j++) {
                uint s = j == 0 ? r.x : r.y;
                if (s == 0xFFFFFFFFu || !face_has(tri, s, v)) continue;
                float2 cu = cuv[3 * s + corner_of(tri, s, v)];
                float2 d = fabs(cu - uvc);
                if (max(d.x, d.y) < 1e-6f) { uvc = cuv[3 * s + corner_of(tri, s, nv)]; break; }
            }
        }
        cuvOut[c] = uvc;
    }
    kernel void face_flag(device const uint* triOut [[buffer(0)]], device uint* keep [[buffer(1)]], constant GP& p [[buffer(15)]],
                          uint f [[thread_position_in_grid]]) {
        if (f > p.b) return;
        keep[f] = (f < p.b && triOut[3 * f] != triOut[3 * f + 1] && triOut[3 * f + 1] != triOut[3 * f + 2] && triOut[3 * f] != triOut[3 * f + 2]) ? 1 : 0;
    }
    kernel void face_scatter(device const uint* triOut [[buffer(0)]], device const float2* cuvOut [[buffer(1)]], device const uint* sc [[buffer(2)]],
                             device uint* tri [[buffer(3)]], device float2* cuv [[buffer(4)]], constant GP& p [[buffer(15)]],
                             uint f [[thread_position_in_grid]]) {
        if (f >= p.b || sc[f + 1] == sc[f]) return;
        uint o = sc[f];
        for (uint k = 0; k < 3; k++) {
            tri[3 * o + k] = triOut[3 * f + k];
            if (p.c & 1) cuv[3 * o + k] = cuvOut[3 * f + k];
        }
    }
    """
}
#endif
