//  Metal source, part 1 of 3, for MeshIO's GPU path (ticket item M5): shared helpers, fill, prefix scan, bounds and
//  the binary-STL kernels. The three parts are concatenated and compiled once by `MeshIOGPU`.
//  Conventions: positions / normals are FLAT float arrays (x y z, 12 B a vertex — not float3's 16 B); counts that
//  a previous kernel produced are read from one small `cnt` buffer so a whole pipeline is ONE command buffer with
//  no CPU read-back in the middle (worst-case sized outputs, the real counts read at the end).

#if canImport(Metal)
extension MeshIOGPU {
    static let sourceCore = """
    #include <metal_stdlib>
    using namespace metal;
    constant uint EMPTY = 0xFFFFFFFFu;

    static inline uint ld32(device const uchar* b, ulong o) {
        return uint(b[o]) | (uint(b[o + 1]) << 8) | (uint(b[o + 2]) << 16) | (uint(b[o + 3]) << 24);
    }
    static inline float ldf(device const uchar* b, ulong o) { return as_type<float>(ld32(b, o)); }
    static inline void st32(device uchar* b, ulong o, uint v) {
        b[o] = uchar(v); b[o + 1] = uchar(v >> 8); b[o + 2] = uchar(v >> 16); b[o + 3] = uchar(v >> 24);
    }
    static inline void stf(device uchar* b, ulong o, float f) { st32(b, o, as_type<uint>(f)); }
    static inline float3 ld3(device const float* p, uint i) { return float3(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
    static inline void st3(device float* p, uint i, float3 v) { p[3 * i] = v.x; p[3 * i + 1] = v.y; p[3 * i + 2] = v.z; }

    kernel void fill_u32(device uint* b [[buffer(0)]], constant uint& n [[buffer(1)]], constant uint& v [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) { if (i < n) b[i] = v; }

    // ---- exclusive prefix sum, three kernels: 1024-wide blocks, block totals, add back ------------------------
    kernel void scan_blocks(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                            device uint* sums [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                            uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup uint part[32];
        uint v = gid < n ? in[gid] : 0;
        uint ex = simd_prefix_exclusive_sum(v);
        uint tot = simd_sum(v);
        if (lane == 0) part[sg] = tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) { uint p = part[lane]; part[lane] = simd_prefix_exclusive_sum(p); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = ex + part[sg];
        if (gid < n) out[gid] = r;
        if (tid == 1023) sums[tg] = r + v;                     // v is 0 past n, so this is the block's total
    }
    kernel void scan_totals(device uint* sums [[buffer(0)]], constant uint& nb [[buffer(1)]],
                            device uint* cnt [[buffer(2)]], constant uint& slot [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
        if (i != 0) return;
        uint run = 0;
        for (uint k = 0; k < nb; k++) { uint v = sums[k]; sums[k] = run; run += v; }
        cnt[slot] = run;
    }
    kernel void scan_add(device uint* out [[buffer(0)]], device const uint* sums [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
        if (gid < n) out[gid] += sums[gid >> 10];
    }

    // ---- bounds: float order-preserving uint keys, one atomic per simdgroup -------------------------------
    static inline uint ordf(float f) { uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
    static inline float unordf(uint u) { return as_type<float>((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u); }

    kernel void bounds_minmax(device const float* P [[buffer(0)]], constant uint& n [[buffer(1)]],
                              device atomic_uint* mm [[buffer(2)]],      // [min xyz, max xyz] ordered keys
                              uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
        float3 lo = float3(INFINITY), hi = float3(-INFINITY);
        if (i < n) { lo = ld3(P, i); hi = lo; }
        lo = float3(simd_min(lo.x), simd_min(lo.y), simd_min(lo.z));
        hi = float3(simd_max(hi.x), simd_max(hi.y), simd_max(hi.z));
        if (lane == 0 && isfinite(lo.x + lo.y + lo.z + hi.x + hi.y + hi.z)) {
            atomic_fetch_min_explicit(&mm[0], ordf(lo.x), memory_order_relaxed);
            atomic_fetch_min_explicit(&mm[1], ordf(lo.y), memory_order_relaxed);
            atomic_fetch_min_explicit(&mm[2], ordf(lo.z), memory_order_relaxed);
            atomic_fetch_max_explicit(&mm[3], ordf(hi.x), memory_order_relaxed);
            atomic_fetch_max_explicit(&mm[4], ordf(hi.y), memory_order_relaxed);
            atomic_fetch_max_explicit(&mm[5], ordf(hi.z), memory_order_relaxed);
        }
    }
    /// The weld quantum MeshIO's CPU path uses: max(1e-9, |bounds diagonal| · 1e-6).
    kernel void weld_quantum(device const uint* mm [[buffer(0)]], device float* q [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
        if (i != 0) return;
        float3 lo = float3(unordf(mm[0]), unordf(mm[1]), unordf(mm[2]));
        float3 hi = float3(unordf(mm[3]), unordf(mm[4]), unordf(mm[5]));
        q[0] = max(1e-9f, length(hi - lo) * 1e-6f);
    }

    // ---- binary STL: 50-byte facets <-> flat float arrays --------------------------------------------------
    /// Decode: positions (9 floats a facet) and the facet normal the CPU's `appendFacet` would keep (stored one
    /// unless zero or disagreeing in sign with the winding's), repeated for the 3 corners.
    kernel void stl_decode(device const uchar* raw [[buffer(0)]], device float* pos [[buffer(1)]],
                           device float* nrm [[buffer(2)]], constant uint& count [[buffer(3)]],
                           uint t [[thread_position_in_grid]]) {
        if (t >= count) return;
        ulong o = 84ul + 50ul * t;
        float3 s = float3(ldf(raw, o), ldf(raw, o + 4), ldf(raw, o + 8));
        float3 a = float3(ldf(raw, o + 12), ldf(raw, o + 16), ldf(raw, o + 20));
        float3 b = float3(ldf(raw, o + 24), ldf(raw, o + 28), ldf(raw, o + 32));
        float3 c = float3(ldf(raw, o + 36), ldf(raw, o + 40), ldf(raw, o + 44));
        float3 r = cross(b - a, c - a); float rl = length(r);
        float3 n = rl > 1e-30f ? r / rl : float3(0, 0, 1);
        float sl = length(s);
        if (sl > 1e-9f && rl > 1e-30f && dot(s / sl, n) > 0) n = s / sl;
        st3(pos, 3 * t, a); st3(pos, 3 * t + 1, b); st3(pos, 3 * t + 2, c);
        st3(nrm, 3 * t, n); st3(nrm, 3 * t + 1, n); st3(nrm, 3 * t + 2, n);
    }
    /// Encode: one facet record (normal from the winding, 3 vertices, 2 zero attribute bytes) per triangle,
    /// written at `out` + 84. Out-of-range indices write a zero facet (the CPU path would trap).
    kernel void stl_encode(device const float* P [[buffer(0)]], device const uint* I [[buffer(1)]],
                           device uchar* out [[buffer(2)]], constant uint& nT [[buffer(3)]],
                           constant uint& nV [[buffer(4)]], uint t [[thread_position_in_grid]]) {
        if (t >= nT) return;
        ulong o = 84ul + 50ul * t;
        uint ia = I[3 * t], ib = I[3 * t + 1], ic = I[3 * t + 2];
        if (ia >= nV || ib >= nV || ic >= nV) { for (uint k = 0; k < 50; k++) out[o + k] = 0; return; }
        float3 a = ld3(P, ia), b = ld3(P, ib), c = ld3(P, ic);
        float3 r = cross(b - a, c - a); float rl = length(r);
        float3 n = rl > 1e-30f ? r / rl : float3(0);
        stf(out, o, n.x); stf(out, o + 4, n.y); stf(out, o + 8, n.z);
        stf(out, o + 12, a.x); stf(out, o + 16, a.y); stf(out, o + 20, a.z);
        stf(out, o + 24, b.x); stf(out, o + 28, b.y); stf(out, o + 32, b.z);
        stf(out, o + 36, c.x); stf(out, o + 40, c.y); stf(out, o + 44, c.z);
        out[o + 48] = 0; out[o + 49] = 0;
    }

    """
}
#endif
