//  Metal source for the GPU-heavy boolean pipeline (`BooleanGPU2`, item M6b). Compiled from source on first use.
//
//  Every kernel works in Float on coordinates relative to a shared origin, so each decision it makes is either
//  conservative (a superset of candidates, or a verdict whose error bound it has checked) or handed back as
//  "uncertain" for the exact double / exact-predicate CPU code. Double-precision values never come from here:
//  the CPU recomputes the few it needs. Kernels (in pipeline order):
//
//    scan_block / scan_add        exclusive prefix sum (1024 per block, recursive on the block sums)
//    tri_boxes                    triangle boxes from the point table
//    grid_count / grid_fill       uniform grid of boxes in CSR form (counting sort by cell)
//    query                        grid query: box overlap / point near edge / point near point, two phases
//                                 (count, then fill + per-query sort so the pair order is deterministic)
//    narrow                       triangle–triangle filter: definite miss / definite hit (with the two segment
//                                 endpoints as edge descriptors) / uncertain
//    compact_nonmiss              stream compaction of the narrow phase's output
//    tri_flags / compact_split    which triangles are cut (constraint or split edge) and the compaction
//    csr_count / csr_fill, cc_*   vertex→face CSR and connected components (hook + pointer jumping) of the faces
//    area_max / area_win          the largest face of every patch
//    winding_tg                   generalized winding numbers, one threadgroup per point
//    asm_*                        keep flags, compaction, vertex remap, barycentric normals/UVs, indices

enum BooleanGPU2Source {
    static let text = """
    #include <metal_stdlib>
    using namespace metal;

    // ---------------------------------------------------------------- scan
    kernel void scan_block(device uint* d [[buffer(0)]], device uint* sums [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           uint gid [[thread_position_in_grid]], uint grp [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint sl [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint ssum[32];
        if (lid < 32) ssum[lid] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint v = gid < n ? d[gid] : 0;
        uint ex = simd_prefix_exclusive_sum(v);
        uint tot = simd_sum(v);
        if (lane == 0) ssum[sl] = tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sl == 0) {
            uint x = ssum[lane];
            uint e = simd_prefix_exclusive_sum(x);
            ssum[lane] = e;
            if (lane == 31) sums[grp] = e + x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (gid < n) d[gid] = ex + ssum[sl];
    }
    kernel void scan_add(device uint* d [[buffer(0)]], device const uint* sums [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint gid [[thread_position_in_grid]], uint grp [[threadgroup_position_in_grid]]) {
        if (gid < n && grp > 0) d[gid] += sums[grp];
    }

    // ---------------------------------------------------------------- boxes
    kernel void tri_boxes(device const uint4* tri [[buffer(0)]], device const float4* pos [[buffer(1)]],
                          device float4* lo [[buffer(2)]], device float4* hi [[buffer(3)]],
                          constant float2& marginN [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= uint(marginN.y)) return;
        uint4 t = tri[i];
        float3 a = pos[t.x].xyz, b = pos[t.y].xyz, c = pos[t.z].xyz;
        lo[i] = float4(min(a, min(b, c)) - marginN.x, 0);
        hi[i] = float4(max(a, max(b, c)) + marginN.x, 0);
    }

    kernel void pt_boxes(device const float4* pos [[buffer(0)]], device float4* lo [[buffer(1)]],
                         device float4* hi [[buffer(2)]], constant float2& marginN [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= uint(marginN.y)) return;
        lo[i] = float4(pos[i].xyz - marginN.x, 0);
        hi[i] = float4(pos[i].xyz + marginN.x, 0);
    }

    // ---------------------------------------------------------------- uniform grid
    struct GridP { float4 lo; float4 inv; uint4 dims; };
    inline int3 cellOf(float3 p, constant GridP& g) {
        float3 q = clamp((p - g.lo.xyz) * g.inv.xyz, float3(-1.0f), float3(g.dims.xyz) + 1.0f);
        return clamp(int3(floor(q)), int3(0), int3(g.dims.xyz) - 1);
    }
    inline uint cellIndex(int3 c, constant GridP& g) { return (uint(c.z) * g.dims.y + uint(c.y)) * g.dims.x + uint(c.x); }

    kernel void grid_count(device const float4* lo [[buffer(0)]], device const float4* hi [[buffer(1)]],
                           constant GridP& g [[buffer(2)]], device atomic_uint* counts [[buffer(3)]],
                           constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        int3 a = cellOf(lo[i].xyz, g), b = cellOf(hi[i].xyz, g);
        int3 d = b - a + 1;
        if (uint(d.x) * uint(d.y) * uint(d.z) > g.dims.w) return;       // too big for the grid: in the big list
        for (int z = a.z; z <= b.z; z++) for (int y = a.y; y <= b.y; y++) for (int x = a.x; x <= b.x; x++)
            atomic_fetch_add_explicit(&counts[cellIndex(int3(x, y, z), g)], 1u, memory_order_relaxed);
    }
    kernel void grid_fill(device const float4* lo [[buffer(0)]], device const float4* hi [[buffer(1)]],
                          constant GridP& g [[buffer(2)]], device const uint* start [[buffer(3)]],
                          device atomic_uint* cursor [[buffer(4)]], device uint* entries [[buffer(5)]],
                          constant uint& n [[buffer(6)]], device uint* big [[buffer(7)]],
                          device atomic_uint* bigCount [[buffer(8)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        int3 a = cellOf(lo[i].xyz, g), b = cellOf(hi[i].xyz, g);
        int3 d = b - a + 1;
        if (uint(d.x) * uint(d.y) * uint(d.z) > g.dims.w) {
            big[atomic_fetch_add_explicit(bigCount, 1u, memory_order_relaxed)] = i;
            return;
        }
        for (int z = a.z; z <= b.z; z++) for (int y = a.y; y <= b.y; y++) for (int x = a.x; x <= b.x; x++) {
            uint c = cellIndex(int3(x, y, z), g);
            uint k = atomic_fetch_add_explicit(&cursor[c], 1u, memory_order_relaxed);
            entries[start[c] + k] = i;
        }
    }

    // mode 0: boxes overlap.  mode 1: Q is a point (id qid) and lies within `margin` of an edge of R's triangle
    // (not one of its corners).  mode 3: Q and R are points; report j < i within `margin`. mode 4: the same for every j.
    struct QP { uint n; uint mode; uint phase; uint cap; float margin; uint nBig; float p1; float p2; };
    inline bool nearEdge(float3 p, uint pid, uint4 t, device const float4* pos, float margin) {
        if (t.x == pid || t.y == pid || t.z == pid) return false;
        for (int k = 0; k < 3; k++) {
            float3 u = pos[t[k]].xyz, v = pos[t[(k + 1) % 3]].xyz;
            float3 ab = v - u;
            float den = dot(ab, ab);
            if (den <= 0) continue;
            float tt = dot(p - u, ab) / den;
            if (tt < -1e-3f || tt > 1.001f) continue;
            float3 q = u + ab * clamp(tt, 0.0f, 1.0f);
            if (length(p - q) <= margin) return true;
        }
        return false;
    }
    inline bool accept(float3 ql, float3 qh, float3 rl, float3 rh, uint i, uint j, constant QP& qp,
                       device const uint4* triR, device const float4* pos) {
        if (rl.x > qh.x || rh.x < ql.x || rl.y > qh.y || rh.y < ql.y || rl.z > qh.z || rh.z < ql.z) return false;
        if (qp.mode == 1) return nearEdge((ql + qh) * 0.5f, i, triR[j], pos, qp.margin);
        if (qp.mode == 3 || qp.mode == 4) {
            if (qp.mode == 3 && j >= i) return false;
            return length(rl - (ql + qh) * 0.5f) <= qp.margin;
        }
        return true;
    }
    kernel void query(device const float4* loQ [[buffer(0)]], device const float4* hiQ [[buffer(1)]],
                      device const float4* loR [[buffer(2)]], device const float4* hiR [[buffer(3)]],
                      device const uint* start [[buffer(4)]], device const uint* entries [[buffer(5)]],
                      constant GridP& g [[buffer(6)]], constant QP& qp [[buffer(7)]],
                      device uint* counts [[buffer(8)]], device const uint* offs [[buffer(9)]],
                      device uint2* out [[buffer(10)]],
                      device const uint* big [[buffer(11)]], device const uint4* triR [[buffer(12)]],
                      device const float4* pos [[buffer(13)]],
                      uint i [[thread_position_in_grid]]) {
        if (i >= qp.n) return;
        float3 ql = loQ[i].xyz, qh = hiQ[i].xyz;
        int3 a = cellOf(ql, g), b = cellOf(qh, g);
        uint count = 0;
        uint base = qp.phase == 1 ? offs[i] : 0;
        for (int z = a.z; z <= b.z; z++) for (int y = a.y; y <= b.y; y++) for (int x = a.x; x <= b.x; x++) {
            uint c = cellIndex(int3(x, y, z), g);
            for (uint e = start[c]; e < start[c + 1]; e++) {
                uint j = entries[e];
                float3 rl = loR[j].xyz, rh = hiR[j].xyz;
                if (cellIndex(cellOf(max(ql, rl), g), g) != c) continue;      // each pair once: in the cell of the overlap's corner
                if (!accept(ql, qh, rl, rh, i, j, qp, triR, pos)) continue;
                if (qp.phase == 1 && count < qp.cap) out[base + count] = uint2(i, j);
                count++;
            }
        }
        for (uint e = 0; e < qp.nBig; e++) {                                    // boxes too big for the grid
            uint j = big[e];
            if (!accept(ql, qh, loR[j].xyz, hiR[j].xyz, i, j, qp, triR, pos)) continue;
            if (qp.phase == 1 && count < qp.cap) out[base + count] = uint2(i, j);
            count++;
        }
        if (qp.phase == 0) { counts[i] = count; return; }
        // insertion sort of this query's pairs by j: the order is then independent of thread scheduling
        for (uint s = 1; s < count; s++) {
            uint2 v = out[base + s];
            uint k = s;
            while (k > 0 && out[base + k - 1].y > v.y) { out[base + k] = out[base + k - 1]; k--; }
            out[base + k] = v;
        }
    }

    // ---------------------------------------------------------------- narrow phase
    struct NP { uint n; float L; float p0; float p1; };
    struct Cross { float pos; float delta; uint code; float3 p; };

    // The point where triangle V's edge (lone -> o) crosses the plane with normal n through w0.
    inline Cross crossing(thread const float3* V, thread const float* d, thread const float* e, uint lone, uint o,
                          float3 D, float marg0) {
        float dl = d[lone], dO = d[o];
        float3 p = V[lone] + (V[o] - V[lone]) * (dl / (dl - dO));
        Cross c;
        c.p = p;
        c.pos = dot(p, D);
        float len = length(V[o] - V[lone]);
        c.delta = len * min(1.0f, 2.0f * max(e[lone], e[o]) / (fabs(dl) + fabs(dO))) + marg0;
        uint k = (o == (lone + 1) % 3) ? lone : o;
        c.code = k << 1;
        return c;
    }

    // status: 0 miss, 1 hit, 2 uncertain. A hit's endpoints are edge descriptors (owner bit 0: the S triangle's
    // edge crossing T's plane, 1: T's edge crossing S's plane; bits 1-2: the edge, k -> k+1) and float points.
    kernel void narrow(device const uint2* pairs [[buffer(0)]], device const uint4* triA [[buffer(1)]],
                       device const uint4* triB [[buffer(2)]], device const float4* pos [[buffer(3)]],
                       constant NP& np [[buffer(4)]], device uint4* outCode [[buffer(5)]],
                       device float4* outEp [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
        if (tid >= np.n) return;
        uint2 pr = pairs[tid];
        uint4 ta = triA[pr.x], tb = triB[pr.y];
        float3 T[3] = { pos[ta.x].xyz, pos[ta.y].xyz, pos[ta.z].xyz };
        float3 S[3] = { pos[tb.x].xyz, pos[tb.y].xyz, pos[tb.z].xyz };
        float L = np.L;
        const float KE = 3e-7f;
        uint status = 2, e0 = 0, e1 = 0, why = 0;
        float3 p0 = float3(0), p1 = float3(0);
        do {
            float3 nT = cross(T[1] - T[0], T[2] - T[0]), nS = cross(S[1] - S[0], S[2] - S[0]);
            float lT = length(nT), lS = length(nS);
            if (lT < 1e-30f || lS < 1e-30f) { why = 1; break; }
            float mT = max(length(T[1] - T[0]), max(length(T[2] - T[0]), length(T[2] - T[1])));
            float mS = max(length(S[1] - S[0]), max(length(S[2] - S[0]), length(S[2] - S[1])));
            float dS[3], eS[3], dT[3], eT[3];
            for (int i = 0; i < 3; i++) {
                float3 w = S[i] - T[0];
                dS[i] = dot(nT, w);
                eS[i] = KE * mT * (length(w) + mT) * (L + 2 * mT);
                float3 u = T[i] - S[0];
                dT[i] = dot(nS, u);
                eT[i] = KE * mS * (length(u) + mS) * (L + 2 * mS);
            }
            if ((dS[0] > eS[0] && dS[1] > eS[1] && dS[2] > eS[2]) || (dS[0] < -eS[0] && dS[1] < -eS[1] && dS[2] < -eS[2])
                || (dT[0] > eT[0] && dT[1] > eT[1] && dT[2] > eT[2]) || (dT[0] < -eT[0] && dT[1] < -eT[1] && dT[2] < -eT[2])) {
                status = 0; break;
            }
            bool clear = true;
            for (int i = 0; i < 3; i++) clear = clear && fabs(dS[i]) > eS[i] && fabs(dT[i]) > eT[i];
            if (!clear) { why = 2; break; }
            float3 D = cross(nT, nS);
            float c = length(D) / (lT * lS);
            if (c < 0.02f) { why = 3; break; }
            D = normalize(D);
            float marg0 = 1e-6f * L / c;
            uint ls = dS[0] > 0 == dS[1] > 0 ? 2 : (dS[0] > 0 == dS[2] > 0 ? 1 : 0);
            uint lt = dT[0] > 0 == dT[1] > 0 ? 2 : (dT[0] > 0 == dT[2] > 0 ? 1 : 0);
            Cross s0 = crossing(S, dS, eS, ls, (ls + 1) % 3, D, marg0), s1 = crossing(S, dS, eS, ls, (ls + 2) % 3, D, marg0);
            Cross t0 = crossing(T, dT, eT, lt, (lt + 1) % 3, D, marg0), t1 = crossing(T, dT, eT, lt, (lt + 2) % 3, D, marg0);
            if (s0.pos > s1.pos) { Cross x = s0; s0 = s1; s1 = x; }
            if (t0.pos > t1.pos) { Cross x = t0; t0 = t1; t1 = x; }
            float U = s0.delta + s1.delta + t0.delta + t1.delta;
            float lo = max(s0.pos, t0.pos), hi = min(s1.pos, t1.pos);
            float ov = hi - lo;
            if (ov < -U) { status = 0; break; }
            if (ov <= U) { why = 4; break; }
            if (fabs(s0.pos - t0.pos) <= s0.delta + t0.delta) { why = 5; break; }
            if (fabs(s1.pos - t1.pos) <= s1.delta + t1.delta) { why = 6; break; }
            bool loS = s0.pos > t0.pos, hiS = s1.pos < t1.pos;
            e0 = loS ? s0.code : (t0.code | 1); p0 = loS ? s0.p : t0.p;
            e1 = hiS ? s1.code : (t1.code | 1); p1 = hiS ? s1.p : t1.p;
            status = 1;
        } while (false);
        outCode[tid] = uint4(status, e0, e1, why);
        outEp[tid * 2] = float4(p0, 0);
        outEp[tid * 2 + 1] = float4(p1, 0);
    }

    // flags[i] = 1 for non-miss pairs; compaction writes (pair index, e0, e1, status) in pair order
    kernel void nonmiss_flags(device const uint4* code [[buffer(0)]], device uint* flags [[buffer(1)]],
                              constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < n) flags[i] = code[i].x != 0 ? 1 : 0;
    }
    kernel void compact_nonmiss(device const uint4* code [[buffer(0)]], device const uint* prefix [[buffer(1)]],
                                device uint4* outCode [[buffer(2)]], device uint* outIdx [[buffer(3)]],
                                constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n || code[i].x == 0) return;
        uint k = prefix[i];
        outCode[k] = code[i];
        outIdx[k] = i;
    }

    // ---------------------------------------------------------------- cut triangles
    inline bool inKeys(device const uint2* keys, uint nKeys, uint a, uint b) {
        uint lo = min(a, b), hi = max(a, b);
        uint l = 0, r = nKeys;
        while (l < r) {
            uint m = (l + r) / 2;
            uint2 k = keys[m];
            if (k.x < lo || (k.x == lo && k.y < hi)) l = m + 1; else r = m;
        }
        return l < nKeys && keys[l].x == lo && keys[l].y == hi;
    }
    kernel void tri_flags(device const uint4* tri [[buffer(0)]], device const uint* consFlag [[buffer(1)]],
                          device const uint2* keys [[buffer(2)]], constant uint2& nn [[buffer(3)]],
                          device uint* flags [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= nn.x) return;
        uint4 t = tri[i];
        bool w = consFlag[i] != 0 || inKeys(keys, nn.y, t.x, t.y) || inKeys(keys, nn.y, t.y, t.z) || inKeys(keys, nn.y, t.z, t.x);
        flags[i] = w ? 1 : 0;
    }
    kernel void compact_split(device const uint* unused [[buffer(0)]], device const uint* prefix [[buffer(1)]],
                              device const uint4* tri [[buffer(2)]], device const uint* srcTri [[buffer(3)]],
                              device uint* work [[buffer(4)]], device uint4* cleanFace [[buffer(5)]],
                              device uint* cleanSrc [[buffer(6)]], constant uint& n [[buffer(7)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        if (prefix[i + 1] != prefix[i]) { work[prefix[i]] = i; return; }
        uint k = i - prefix[i];
        cleanFace[k] = tri[i];
        cleanSrc[k] = srcTri[i];
    }

    // ---------------------------------------------------------------- face patches
    kernel void csr_count(device const uint4* faces [[buffer(0)]], device atomic_uint* counts [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint4 f = faces[i];
        atomic_fetch_add_explicit(&counts[f.x], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&counts[f.y], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&counts[f.z], 1u, memory_order_relaxed);
    }
    kernel void csr_fill(device const uint4* faces [[buffer(0)]], device const uint* start [[buffer(1)]],
                         device atomic_uint* cursor [[buffer(2)]], device uint* list [[buffer(3)]],
                         constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint4 f = faces[i];
        for (int k = 0; k < 3; k++) {
            uint v = f[k];
            list[start[v] + atomic_fetch_add_explicit(&cursor[v], 1u, memory_order_relaxed)] = i;
        }
    }
    kernel void cc_init(device uint* parent [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        if (i < n) parent[i] = i;
    }
    // faces sharing an uncut edge are one patch; labels only ever decrease, so the fixpoint labels a patch by its
    // lowest face
    kernel void cc_hook(device const uint4* faces [[buffer(0)]], device const uint* start [[buffer(1)]],
                        device const uint* list [[buffer(2)]], device const uint2* cutKeys [[buffer(3)]],
                        constant uint2& nn [[buffer(4)]], device atomic_uint* parent [[buffer(5)]],
                        device atomic_uint* changed [[buffer(6)]], uint f [[thread_position_in_grid]]) {
        if (f >= nn.x) return;
        uint4 fc = faces[f];
        for (int k = 0; k < 3; k++) {
            uint a = fc[k], b = fc[(k + 1) % 3];
            if (inKeys(cutKeys, nn.y, a, b)) continue;
            for (uint e = start[a]; e < start[a + 1]; e++) {
                uint g = list[e];
                if (g == f) continue;
                uint4 gc = faces[g];
                if (gc.x != b && gc.y != b && gc.z != b) continue;
                uint pf = atomic_load_explicit(&parent[f], memory_order_relaxed);
                uint pg = atomic_load_explicit(&parent[g], memory_order_relaxed);
                if (pf == pg) continue;
                atomic_fetch_min_explicit(&parent[max(pf, pg)], min(pf, pg), memory_order_relaxed);
                atomic_store_explicit(changed, 1u, memory_order_relaxed);
            }
        }
    }
    kernel void cc_jump(device atomic_uint* parent [[buffer(0)]], device atomic_uint* changed [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint p = atomic_load_explicit(&parent[i], memory_order_relaxed);
        uint pp = atomic_load_explicit(&parent[p], memory_order_relaxed);
        if (pp != p) { atomic_store_explicit(&parent[i], pp, memory_order_relaxed); atomic_store_explicit(changed, 1u, memory_order_relaxed); }
    }
    inline float faceArea(uint4 f, device const float4* pos) {
        float3 a = pos[f.x].xyz;
        return 0.5f * length(cross(pos[f.y].xyz - a, pos[f.z].xyz - a));
    }
    kernel void area_max(device const uint4* faces [[buffer(0)]], device const float4* pos [[buffer(1)]],
                         device const uint* parent [[buffer(2)]], device atomic_uint* maxA [[buffer(3)]],
                         constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        atomic_fetch_max_explicit(&maxA[parent[i]], as_type<uint>(faceArea(faces[i], pos)), memory_order_relaxed);
    }
    kernel void area_win(device const uint4* faces [[buffer(0)]], device const float4* pos [[buffer(1)]],
                         device const uint* parent [[buffer(2)]], device const uint* maxA [[buffer(3)]],
                         device atomic_uint* win [[buffer(4)]], constant uint& n [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        if (as_type<uint>(faceArea(faces[i], pos)) == maxA[parent[i]])
            atomic_fetch_min_explicit(&win[parent[i]], i, memory_order_relaxed);
    }
    kernel void root_flags(device const uint* parent [[buffer(0)]], device uint* flags [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < n) flags[i] = parent[i] == i ? 1 : 0;
    }
    kernel void compact_roots(device const uint* parent [[buffer(0)]], device const uint* prefix [[buffer(1)]],
                              device const uint* win [[buffer(2)]], device uint2* out [[buffer(3)]],
                              constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i < n && parent[i] == i) out[prefix[i]] = uint2(i, win[i]);
    }

    // ---------------------------------------------------------------- winding numbers
    kernel void winding_tg(device const float4* pts [[buffer(0)]], device const uint4* tri [[buffer(1)]],
                           device const float4* pos [[buffer(2)]], device float* out [[buffer(3)]],
                           constant uint& nTri [[buffer(4)]],
                           uint grp [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                           uint sl [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float part[8];
        float3 p = pts[grp].xyz;
        float sum = 0;
        for (uint t = lid; t < nTri; t += 256) {
            uint4 v = tri[t];
            float3 a = pos[v.x].xyz - p, b = pos[v.y].xyz - p, c = pos[v.z].xyz - p;
            float la = length(a), lb = length(b), lc = length(c);
            float num = dot(a, cross(b, c));
            float den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
            sum += 2.0f * atan2(num, den);
        }
        sum = simd_sum(sum);
        if (lane == 0) part[sl] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float s = 0;
            for (int k = 0; k < 8; k++) s += part[k];
            out[grp] = s / (4.0f * M_PI_F);
        }
    }

    // ---------------------------------------------------------------- assembly
    // rel codes: 0 inside, 1 outside, 2 same, 3 opposite.  op: 0 union, 1 difference, 2 intersection.
    struct AP { uint nFaces; uint nFacesA; uint op; uint nPool; uint nVertA; uint idxOffB; uint hasN; uint hasUV; };
    inline uint keepFlip(uint op, uint side, uint rel) {      // bit0 keep, bit1 flip
        if (op == 0) return (rel == 1 || (side == 0 && rel == 2)) ? 1 : 0;
        if (op == 2) return (rel == 0 || (side == 0 && rel == 2)) ? 1 : 0;
        if (side == 0) return (rel == 1 || rel == 3) ? 1 : 0;
        return rel == 0 ? 3 : 0;
    }
    kernel void asm_flags(device const uint4* faces [[buffer(0)]], device const uint* parent [[buffer(1)]],
                          device const uint* relOfRoot [[buffer(2)]], constant AP& ap [[buffer(3)]],
                          device uint* flags [[buffer(4)]], device const uint* parent1 [[buffer(5)]],
                          device const uint* rel1 [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        if (i >= ap.nFaces) return;
        uint4 f = faces[i];
        uint side = i < ap.nFacesA ? 0 : 1;
        uint kf = keepFlip(ap.op, side, side == 0 ? relOfRoot[parent[i]] : rel1[parent1[i - ap.nFacesA]]);
        bool degenerate = f.x == f.y || f.y == f.z || f.x == f.z;
        flags[i] = (kf & 1) != 0 && !degenerate ? 1 : 0;
    }
    // kept faces in order: pool corners (flip applied to the winding), source face, flags; mark pool vertices used
    kernel void asm_compact(device const uint4* faces [[buffer(0)]], device const uint* parent [[buffer(1)]],
                            device const uint* relOfRoot [[buffer(2)]], constant AP& ap [[buffer(3)]],
                            device const uint* flags [[buffer(4)]], device const uint* prefix [[buffer(5)]],
                            device uint4* kept [[buffer(6)]], device uint* keptFace [[buffer(7)]],
                            device uint* used [[buffer(8)]], device const uint* parent1 [[buffer(9)]],
                            device const uint* rel1 [[buffer(10)]], uint i [[thread_position_in_grid]]) {
        if (i >= ap.nFaces || prefix[i + 1] == prefix[i]) return;
        uint4 f = faces[i];
        uint side = i < ap.nFacesA ? 0 : 1;
        uint kf = keepFlip(ap.op, side, side == 0 ? relOfRoot[parent[i]] : rel1[parent1[i - ap.nFacesA]]);
        uint k = prefix[i];
        kept[k] = (kf & 2) != 0 ? uint4(f.x, f.z, f.y, kf & 2) : uint4(f.x, f.y, f.z, 0);
        keptFace[k] = i;
        used[f.x] = 1; used[f.y] = 1; used[f.z] = 1;
    }
    kernel void asm_used_list(device const uint* used [[buffer(0)]], device const uint* prefix [[buffer(1)]],
                              device uint* list [[buffer(2)]], constant uint& n [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
        if (i < n && used[i] != 0) list[prefix[i]] = i;
    }
    inline float3 hl(device const float4* hi, device const float4* lo, uint i) { return hi[i].xyz + lo[i].xyz; }

    // Per kept corner: barycentrics of the corner inside its source triangle (double-single arithmetic: the
    // pool and source positions are hi + lo floats), the interpolated normal (fixed-point atomic sums, so the
    // total is the same on every run) and UV; the first corner to claim a vertex gives its UV.
    kernel void asm_corners(device const uint4* kept [[buffer(0)]], device const uint* keptFace [[buffer(1)]],
                            device const uint* faceSrc [[buffer(2)]], constant AP& ap [[buffer(3)]],
                            device const uint* srcIdx [[buffer(4)]],
                            device const float4* srcHi [[buffer(5)]], device const float4* srcLo [[buffer(6)]],
                            device const float4* srcN [[buffer(7)]], device const float2* srcUV [[buffer(8)]],
                            device const float4* poolHi [[buffer(9)]], device const float4* poolLo [[buffer(10)]],
                            device const uint* newIndex [[buffer(11)]],
                            device atomic_int* nsum [[buffer(12)]], device atomic_uint* firstCorner [[buffer(13)]],
                            device float2* cornerUV [[buffer(14)]], device uint* outIdx [[buffer(15)]],
                            constant uint& nKept [[buffer(16)]], uint k [[thread_position_in_grid]]) {
        if (k >= nKept) return;
        uint4 f = kept[k];
        uint fi = keptFace[k];
        uint side = fi < ap.nFacesA ? 0 : 1;
        uint vOff = side == 0 ? 0 : ap.nVertA;
        uint src = faceSrc[fi];
        uint ib = (side == 0 ? 0 : ap.idxOffB) + src * 3;
        uint i0 = srcIdx[ib] + vOff, i1 = srcIdx[ib + 1] + vOff, i2 = srcIdx[ib + 2] + vOff;
        float3 a = hl(srcHi, srcLo, i0), b = hl(srcHi, srcLo, i1), c = hl(srcHi, srcLo, i2);
        float3 v0 = (srcHi[i1].xyz - srcHi[i0].xyz) + (srcLo[i1].xyz - srcLo[i0].xyz);
        float3 v1 = (srcHi[i2].xyz - srcHi[i0].xyz) + (srcLo[i2].xyz - srcLo[i0].xyz);
        float3 fn = normalize(cross(v0, v1));
        float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
        float den = d00 * d11 - d01 * d01;
        uint ids[3] = { f.x, f.y, f.z };
        // `kept` has the swapped winding when flipped; the corners are processed in the ORIGINAL face order
        bool flip = f.w != 0;
        for (uint cidx = 0; cidx < 3; cidx++) {
            uint pid = ids[cidx];
            float3 v2 = (poolHi[pid].xyz - srcHi[i0].xyz) + (poolLo[pid].xyz - srcLo[i0].xyz);
            float d20 = dot(v2, v0), d21 = dot(v2, v1);
            float bv = den != 0 ? (d11 * d20 - d01 * d21) / den : 0;
            float bw = den != 0 ? (d00 * d21 - d01 * d20) / den : 0;
            float bu = 1 - bv - bw;
            float3 n = fn;
            if (((ap.hasN >> side) & 1) != 0) {
                float3 nn = srcN[i0].xyz * bu + srcN[i1].xyz * bv + srcN[i2].xyz * bw;
                float ln = length(nn);
                n = ln > 0 ? nn / ln : fn;
            }
            if (flip) n = -n;
            float2 uv = ((ap.hasUV >> side) & 1) != 0 ? srcUV[i0] * bu + srcUV[i1] * bv + srcUV[i2] * bw : float2(0);
            uint v = newIndex[pid];
            atomic_fetch_add_explicit(&nsum[v * 4 + 0], int(rint(n.x * 16777216.0f)), memory_order_relaxed);
            atomic_fetch_add_explicit(&nsum[v * 4 + 1], int(rint(n.y * 16777216.0f)), memory_order_relaxed);
            atomic_fetch_add_explicit(&nsum[v * 4 + 2], int(rint(n.z * 16777216.0f)), memory_order_relaxed);
            uint corner = k * 3 + cidx;
            cornerUV[corner] = uv;
            atomic_fetch_min_explicit(&firstCorner[v], corner, memory_order_relaxed);
        }
        uint o = k * 3;
        // output winding: kept[k] is already swapped for a flip, matching the CPU's idx.swapAt(1, 2)
        outIdx[o] = newIndex[f.x]; outIdx[o + 1] = newIndex[f.y]; outIdx[o + 2] = newIndex[f.z];
    }
    kernel void asm_vertices(device const int* nsum [[buffer(0)]], device const uint* firstCorner [[buffer(1)]],
                             device const float2* cornerUV [[buffer(2)]], device float4* outN [[buffer(3)]],
                             device float2* outUV [[buffer(4)]], constant uint& n [[buffer(5)]],
                             uint v [[thread_position_in_grid]]) {
        if (v >= n) return;
        float3 s = float3(nsum[v * 4], nsum[v * 4 + 1], nsum[v * 4 + 2]);
        float l = length(s);
        outN[v] = float4(l > 0 ? s / l : float3(0, 0, 1), 0);
        uint c = firstCorner[v];
        outUV[v] = c == 0xFFFFFFFFu ? float2(0) : cornerUV[c];
    }
    """
}
