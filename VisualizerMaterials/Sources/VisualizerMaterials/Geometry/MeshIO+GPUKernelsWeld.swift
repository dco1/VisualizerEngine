//  Metal source, part 2 of 3, for MeshIO's GPU path: grid-hash weld, angle-weighted crease-smoothed normals and
//  per-triangle winding / volume checks.
//
//  Weld: every vertex quantises to an integer cell (round(p / q), the CPU weld's own rule) and inserts itself into an
//  open-addressing table of vertex indices (CAS on a 32-bit slot; keys are re-derived from the occupant's position,
//  so no 64-bit atomics). Each group's LOWEST index is its representative, so the output order is the CPU's
//  first-occurrence order; survivors are numbered by a prefix scan.
//  Smooth: a vertex's normal is the angle-weighted sum of the incident faces within the crease angle of the face that
//  touched it LAST (`Mesh3.smoothed`'s write order), so the pass is: lastTri by atomic max, then per-corner float
//  atomic adds. The one deviation from the CPU: float atomics add in no fixed order (~1e-7 differences run to run).

#if canImport(Metal)
extension MeshIOGPU {
    static let sourceWeld = """
    static inline uint mix32(uint h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
    static inline uint hash3(int3 k) { return mix32(uint(k.x) * 0x9E3779B1u ^ mix32(uint(k.y) * 0x85EBCA77u ^ mix32(uint(k.z) * 0xC2B2AE3Du))); }
    static inline int3 qkey(float3 p, float q) { return int3(round(p / q)); }

    kernel void weld_insert(device const float* P [[buffer(0)]], constant uint& n [[buffer(1)]],
                            device const float* qb [[buffer(2)]], device atomic_uint* slots [[buffer(3)]],
                            device uint* rep [[buffer(4)]], constant uint& mask [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        float q = qb[0]; int3 k = qkey(ld3(P, i), q);
        uint s = hash3(k) & mask;
        for (;;) {
            uint expected = EMPTY;
            if (atomic_compare_exchange_weak_explicit(&slots[s], &expected, i, memory_order_relaxed, memory_order_relaxed)) { rep[i] = i; return; }
            if (expected == EMPTY) continue;                       // spurious failure: same slot again
            if (all(qkey(ld3(P, expected), q) == k)) { rep[i] = expected; return; }
            s = (s + 1) & mask;
        }
    }
    kernel void weld_groupmin(device const uint* rep [[buffer(0)]], device atomic_uint* gmin [[buffer(1)]],
                              constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < n) atomic_fetch_min_explicit(&gmin[rep[i]], i, memory_order_relaxed);
    }
    /// canon[i] = group's lowest index; flag[i] = 1 where i survives.
    kernel void weld_canon(device const uint* rep [[buffer(0)]], device const uint* gmin [[buffer(1)]],
                           device uint* canon [[buffer(2)]], device uint* flag [[buffer(3)]],
                           constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint c = gmin[rep[i]]; canon[i] = c; flag[i] = (c == i) ? 1u : 0u;
    }
    /// Survivors copy their position to their scan slot; every vertex learns its new index in `remap`.
    kernel void weld_emit(device const float* P [[buffer(0)]], device const uint* canon [[buffer(1)]],
                          device const uint* scan [[buffer(2)]], device float* outP [[buffer(3)]],
                          device uint* remap [[buffer(4)]], constant uint& n [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint c = canon[i]; uint ni = scan[c]; remap[i] = ni;
        if (c == i) st3(outP, ni, ld3(P, i));
    }
    /// Triangle (t) through the remap (identity index when `I` is null: STL's 3t+k); a weld that collapses a
    /// triangle to a line or point drops it.
    kernel void weld_tri_flags(device const uint* I [[buffer(0)]], device const uint* remap [[buffer(1)]],
                               device uint* flag [[buffer(2)]], device uint* ra [[buffer(3)]],
                               constant uint& nT [[buffer(4)]], constant uint& identity [[buffer(5)]],
                               uint t [[thread_position_in_grid]]) {
        if (t >= nT) return;
        uint a = remap[identity ? 3 * t : I[3 * t]], b = remap[identity ? 3 * t + 1 : I[3 * t + 1]], c = remap[identity ? 3 * t + 2 : I[3 * t + 2]];
        ra[3 * t] = a; ra[3 * t + 1] = b; ra[3 * t + 2] = c;
        flag[t] = (a != b && b != c && a != c) ? 1u : 0u;
    }
    kernel void weld_tri_compact(device const uint* ra [[buffer(0)]], device const uint* flag [[buffer(1)]],
                                 device const uint* scan [[buffer(2)]], device uint* outI [[buffer(3)]],
                                 constant uint& nT [[buffer(4)]], uint t [[thread_position_in_grid]]) {
        if (t >= nT || flag[t] == 0) return;
        uint o = scan[t];
        outI[3 * o] = ra[3 * t]; outI[3 * o + 1] = ra[3 * t + 1]; outI[3 * o + 2] = ra[3 * t + 2];
    }

    // ---- normals ------------------------------------------------------------------------------------------
    /// Face normals (0 for a degenerate face) and, per vertex, the last (highest) valid triangle + 1.
    kernel void sm_faces(device const float* P [[buffer(0)]], device const uint* I [[buffer(1)]],
                         device const uint* cnt [[buffer(2)]], constant uint& slotT [[buffer(3)]],
                         device float* fn [[buffer(4)]], device atomic_uint* last [[buffer(5)]],
                         uint t [[thread_position_in_grid]]) {
        if (t >= cnt[slotT]) return;
        uint ia = I[3 * t], ib = I[3 * t + 1], ic = I[3 * t + 2];
        float3 a = ld3(P, ia);
        float3 r = cross(ld3(P, ib) - a, ld3(P, ic) - a); float l = length(r);
        if (!(l > 1e-12f)) { st3(fn, t, float3(0)); return; }
        st3(fn, t, r / l);
        atomic_fetch_max_explicit(&last[ia], t + 1, memory_order_relaxed);
        atomic_fetch_max_explicit(&last[ib], t + 1, memory_order_relaxed);
        atomic_fetch_max_explicit(&last[ic], t + 1, memory_order_relaxed);
    }
    kernel void sm_accum(device const float* P [[buffer(0)]], device const uint* I [[buffer(1)]],
                         device const uint* cnt [[buffer(2)]], constant uint& slotT [[buffer(3)]],
                         device const float* fn [[buffer(4)]], device const uint* last [[buffer(5)]],
                         device atomic<float>* acc [[buffer(6)]], constant float& cosCrease [[buffer(7)]],
                         uint t [[thread_position_in_grid]]) {
        if (t >= cnt[slotT]) return;
        float3 f = ld3(fn, t);
        if (all(f == float3(0))) return;
        uint idx[3] = { I[3 * t], I[3 * t + 1], I[3 * t + 2] };
        float3 p[3] = { ld3(P, idx[0]), ld3(P, idx[1]), ld3(P, idx[2]) };
        for (uint k = 0; k < 3; k++) {
            float3 u = p[(k + 1) % 3] - p[k], v = p[(k + 2) % 3] - p[k];
            float lu = length(u), lv = length(v);
            if (!(lu > 1e-12f && lv > 1e-12f)) continue;
            float ang = acos(clamp(dot(u, v) / (lu * lv), -1.0f, 1.0f));
            if (!(ang > 1e-9f)) continue;
            uint i = idx[k];
            float3 ref = ld3(fn, last[i] - 1);                     // the face that wrote this vertex last
            if (dot(ref, f) >= cosCrease) {
                atomic_fetch_add_explicit(&acc[3 * i], f.x * ang, memory_order_relaxed);
                atomic_fetch_add_explicit(&acc[3 * i + 1], f.y * ang, memory_order_relaxed);
                atomic_fetch_add_explicit(&acc[3 * i + 2], f.z * ang, memory_order_relaxed);
            }
        }
    }
    kernel void sm_finish(device const float* acc [[buffer(0)]], device const float* fn [[buffer(1)]],
                          device const uint* last [[buffer(2)]], device float* nrm [[buffer(3)]],
                          device const uint* cnt [[buffer(4)]], constant uint& slotV [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
        if (i >= cnt[slotV]) return;
        uint l = last[i];
        if (l == 0) { st3(nrm, i, float3(0, 0, 1)); return; }
        float3 a = ld3(acc, i); float len = length(a);
        st3(nrm, i, len > 1e-12f ? a / len : ld3(fn, l - 1));
    }
    /// Flat per-vertex normals from faces (`faceNormalsPerVertex`: the last face to touch a vertex wins).
    kernel void nrm_from_last(device const float* fn [[buffer(0)]], device const uint* last [[buffer(1)]],
                              device float* nrm [[buffer(2)]], constant uint& nV [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= nV) return;
        uint l = last[i];
        st3(nrm, i, l == 0 ? float3(0, 0, 1) : ld3(fn, l - 1));
    }

    // ---- checks ---------------------------------------------------------------------------------------------
    /// Per 256-thread group: signed volume (about `org`, for float accuracy), area, degenerate and flipped counts.
    kernel void tri_check(device const float* P [[buffer(0)]], device const uint* I [[buffer(1)]],
                          device const float* N [[buffer(2)]], device const uint* cnt [[buffer(3)]],
                          constant uint& slotT [[buffer(4)]], constant float4& org [[buffer(5)]],
                          constant uint& hasN [[buffer(6)]], device float4* pf [[buffer(7)]], device uint4* pu [[buffer(8)]],
                          uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                          uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup float4 sf[8]; threadgroup uint4 su[8];
        float vol = 0, area = 0; uint deg = 0, flip = 0;
        if (t < cnt[slotT]) {
            uint ia = I[3 * t], ib = I[3 * t + 1], ic = I[3 * t + 2];
            float3 o = org.xyz, a = ld3(P, ia), b = ld3(P, ib), c = ld3(P, ic);
            vol = dot(a - o, cross(b - o, c - o)) / 6.0f;
            float3 r = cross(b - a, c - a); float l = length(r);
            area = 0.5f * l;
            if (!(l > 1e-12f)) deg = 1;
            else if (hasN != 0 && dot(ld3(N, ia) + ld3(N, ib) + ld3(N, ic), r) < 0) flip = 1;
        }
        float v = simd_sum(vol), ar = simd_sum(area); uint d = simd_sum(deg), fl = simd_sum(flip);
        if (lane == 0) { sf[sg] = float4(v, ar, 0, 0); su[sg] = uint4(d, fl, 0, 0); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float4 f = 0; uint4 u = 0;
            for (uint k = 0; k < 8; k++) { f += sf[k]; u += su[k]; }
            pf[tg] = f; pu[tg] = u;
        }
    }
    /// Directed-edge table: each (a -> b) inserts its corner id; a second copy of the same directed edge is a winding
    /// conflict (stats[0]).
    static inline uint2 dedge(device const uint* I, uint e) { return uint2(I[e], I[3 * (e / 3) + (e % 3 + 1) % 3]); }
    kernel void edge_insert(device const uint* I [[buffer(0)]], device const uint* cnt [[buffer(1)]],
                            constant uint& slotT [[buffer(2)]], device atomic_uint* slots [[buffer(3)]],
                            constant uint& mask [[buffer(4)]], device atomic_uint* stats [[buffer(5)]],
                            uint e [[thread_position_in_grid]]) {
        if (e >= 3 * cnt[slotT]) return;
        uint2 ab = dedge(I, e);
        if (ab.x == ab.y) { atomic_fetch_add_explicit(&stats[2], 1u, memory_order_relaxed); return; }
        uint s = mix32(ab.x * 0x9E3779B1u ^ mix32(ab.y)) & mask;
        for (;;) {
            uint expected = EMPTY;
            if (atomic_compare_exchange_weak_explicit(&slots[s], &expected, e, memory_order_relaxed, memory_order_relaxed)) return;
            if (expected == EMPTY) continue;
            if (all(dedge(I, expected) == ab)) { atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed); return; }
            s = (s + 1) & mask;
        }
    }
    /// Each directed edge looks for its reverse; not found = an open (boundary or inconsistently wound) edge (stats[1]).
    kernel void edge_reverse(device const uint* I [[buffer(0)]], device const uint* cnt [[buffer(1)]],
                             constant uint& slotT [[buffer(2)]], device const uint* slots [[buffer(3)]],
                             constant uint& mask [[buffer(4)]], device atomic_uint* stats [[buffer(5)]],
                             uint e [[thread_position_in_grid]]) {
        if (e >= 3 * cnt[slotT]) return;
        uint2 ab = dedge(I, e);
        if (ab.x == ab.y) return;
        uint2 ba = uint2(ab.y, ab.x);
        uint s = mix32(ba.x * 0x9E3779B1u ^ mix32(ba.y)) & mask;
        for (;;) {
            uint r = slots[s];
            if (r == EMPTY) { atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed); return; }
            if (all(dedge(I, r) == ba)) return;
            s = (s + 1) & mask;
        }
    }

    """
}
#endif
